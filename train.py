import argparse
import os
import tempfile
import time
import typing
import torch
import torch.nn
import torch.optim
import copy
import mlflow
import optuna
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import get_logger
from utils.config import load_config
from utils.mlflow_utils import setup_mlflow, log_params, log_metrics, log_artifact, log_config_as_artifact, log_model_summary
from evaluation.statistical_metrics import MetricRegistry
from utils.plot_utils import save_time_series_plots, save_synthetic_data
from data.data_loader import get_dataloader
from architecture.model import build_model
from architecture.flow_matching import FlowMatcher
from utils.schedulers import build_scheduler
from utils.ema import EMA
from utils.loss import build_loss, BaseLoss

logger = get_logger("train")

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    matcher: FlowMatcher,
    device: torch.device,
    config: dict,
    loss_fn: BaseLoss,
    epoch: int,
    ema: typing.Optional[EMA] = None
) -> float:

    model.train()
    total_loss = 0.0
    total_loss_raw = 0.0
    total_loss_fft = 0.0

    fft_weight = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        x_1 = batch.to(device)
        optimizer.zero_grad()

        # 1. Sample and Compute
        x_0 = matcher.sample_noise(x_1.shape, device)
        t = matcher.sample_time(x_1.shape[0], device)
        x_t, u_t = matcher.compute_target(x_0, x_1, t)

        # 2. Forward and Loss
        v_pred = model(x_t, t)
        loss_dict = loss_fn(
            model=model,
            x_t=x_t,
            x_1=x_1,
            u_t=u_t,
            v_pred=v_pred,
            t=t,
            epoch=epoch
        )

        loss = loss_dict["loss"]

        # 3. Step
        loss.backward()

        training_cfg = config.get("training", {})
        gradient_clip = training_cfg.get("gradient_clip", 1.0)
        if gradient_clip is not None and str(gradient_clip).lower() != "none":
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))

        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        total_loss_raw += loss_dict["loss_raw"].item()
        total_loss_fft += loss_dict["loss_fft"].item()
        fft_weight = loss_dict["fft_weight"]

        n = len(dataloader)

    return {
        "train_loss": total_loss / n,
        "train_loss_raw": total_loss_raw / n,
        "train_loss_fft": total_loss_fft / n,
        "fft_weight": fft_weight,
    }
    
@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    matcher: FlowMatcher,
    metrics_registry: MetricRegistry,
    device: torch.device,
    epoch: int,
    config: typing.Optional[dict] = None
) -> dict[str, float]:
    model.eval()
    all_real = []
    all_fake = []

    # Process batches
    for batch in tqdm(dataloader, desc="Validating"):
        x_1 = batch.to(device)
        B, L, F = x_1.shape

        # Euler Integration
        dt = 1.0 / 100
        x = matcher.sample_noise(x_1.shape, device)
        
        for step in range(100):
            t = torch.ones((B,), device=device) * (step * dt)
            v = model(x, t)
            x = x + v * dt  # Basic Euler step

        all_real.append(x_1.cpu())
        all_fake.append(x.cpu())

        # Limit validation size for speed
        if len(all_fake) * B >= 512: 
            break

    # Prepare data for metrics and plotting
    t0 = time.perf_counter()
    torch.cuda.synchronize(device)
    real_cat = torch.cat(all_real, dim=0)
    fake_cat = torch.cat(all_fake, dim=0)
    torch.cuda.synchronize(device)

    # Detect if FFT / frequency features are present
    num_bins = 0
    if hasattr(dataloader.dataset, "num_bins") and dataloader.dataset.num_bins > 0:
        num_bins = dataloader.dataset.num_bins
    elif config is not None:
        dataset_cfg = config.get("dataset", {})
        model_cfg = config.get("model", {})
        num_bins = dataset_cfg.get("num_bins", dataset_cfg.get("number_bins", 0))
        if not num_bins and model_cfg.get("architecture") == "mamba_3_stft":
            window_size = dataset_cfg.get("window_size", 64)
            num_bins = window_size // 2 + 1

    # When FFT features are used, keep only time features using stride = 1 + num_bins
    if num_bins > 0:
        stride = 1 + num_bins
        real_cat_eval = real_cat[:, :, ::stride]
        fake_cat_eval = fake_cat[:, :, ::stride]
    else:
        real_cat_eval = real_cat
        fake_cat_eval = fake_cat

    t0 = time.perf_counter()
    torch.cuda.synchronize(device)
    eval_metrics = metrics_registry.evaluate(real_cat_eval, fake_cat_eval)
    torch.cuda.synchronize(device)

    # Visualizatio
    save_time_series_plots(real_cat_eval.numpy(), fake_cat_eval.numpy(), epoch)
    save_synthetic_data(fake_cat_eval.numpy(), epoch)

    return eval_metrics


def train_trial(
    trial: optuna.Trial,
    config: dict[str, typing.Any],
    device: torch.device,
    use_optuna: bool = True
) -> float:
    """
    Executes a single training run for an Optuna trial.
    Keeps training logic, validation logic, checkpointing, and MLflow logging intact.
    """
    setup_mlflow(config)

    model = build_model(config).to(device)
    train_loader = get_dataloader(config, split="train" )
    train_stats = train_loader.dataset.stats
    val_loader = get_dataloader(config,  split="val", stats=train_stats)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config.get("training", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-5))
    )

    # Build scheduler
    scheduler = build_scheduler(optimizer, config)

    # Create EMA helper
    use_ema = config.get("training", {}).get("use_ema", True)
    if use_ema:
        ema_decay = float(config.get("training", {}).get("ema_decay", 0.999))
        ema = EMA(model, decay=ema_decay)
    else:
        ema = None

    scaler = torch.amp.GradScaler('cuda', enabled=config.get("training", {}).get("mixed_precision", True))

    sigma_min = config.get("flow_matching", {}).get("sigma_min", 1e-4)
    matcher = FlowMatcher(sigma_min=sigma_min)
    loss_fn = build_loss(config)
    metrics_registry = MetricRegistry()

    epochs = config.get("training", {}).get("epochs", 100)
    best_fid = float('inf')
    best_metrics = {}
    run_name = f"trial_{trial.number}" if use_optuna else "single_run"
    with mlflow.start_run(run_name=run_name):
        log_params(config)
        log_config_as_artifact(config)
        log_model_summary(model)

        try:
            for epoch in range(1, epochs + 1):
                start_time = time.time()
                logger.info(f"Starting Epoch {epoch}/{epochs}")

                train_metrics = train_epoch(model, train_loader, optimizer, scaler, matcher, device, config, loss_fn, epoch, ema)

                # Step scheduler immediately after each epoch
                if scheduler is not None:
                    scheduler.step()

                log_dict = {
                        **train_metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                }

                if epoch % config.get("training", {}).get("val_every", 1) == 0:
                    
                    val_model = ema.ema_model if ema is not None else model
                    val_metrics = validate_epoch(val_model, val_loader, matcher, metrics_registry, device, epoch, config)
                    log_dict.update(val_metrics)

                epoch_time = time.time() - start_time
                log_dict["epoch_runtime"] = epoch_time

                log_metrics(log_dict, step=epoch)
                logger.info(f"Epoch {epoch} summary: {log_dict}")

                # Save checkpoint
                current_fid = log_dict.get("frechet_inception_distance")
                is_best = False

                if current_fid is not None:
                    if current_fid < best_fid:
                        best_fid = current_fid
                        is_best = True
                        logger.info(f"New best FID: {best_fid:.4f} at epoch {epoch}")
                        best_metrics = {f"best_{k}": v for k, v in log_dict.items()}
                        best_metrics["best_epoch"] = epoch
                    if use_optuna:
                        trial.report(current_fid, epoch//config.get("training").get("val_every"))

                        if trial.should_prune():
                            raise optuna.TrialPruned()

                if is_best:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Save best model
                        ckpt_path = os.path.join(tmpdir, f"best_model_epoch_{epoch}.pt")

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': ema.state_dict() if ema is not None else model.state_dict(),
                            'ema_state_dict': ema.state_dict() if ema is not None else None,
                            'raw_model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'fid_score': current_fid,
                            'best_fid': best_fid,
                        }, ckpt_path)

                        log_artifact(ckpt_path, "checkpoints")

        finally:
            if best_metrics:
                log_metrics(best_metrics)

    return best_fid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config", default="configs/ett_336_best.yaml")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="MTS-Flow_optuna", help="Optuna study name")
    parser.add_argument("--storage", type=str, default="sqlite:///MTS-Flow_optuna.db", help="Optuna SQLite storage URI")
    parser.add_argument("--use_optuna", type=bool, default=False, help="Whether to use Optuna for hyperparameter optimization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)

    if not args.use_optuna:
        logger.info("Optuna disabled. Executing single training run...")
        train_trial(trial=None, config=config, device=device, use_optuna=args.use_optuna)
        logger.info("Single training run completed.")

    else:
        # Create the Optuna study with:
        # - direction="minimize"
        # - persistent SQLite storage
        # - load_if_exists=True
        # - MedianPruner
        pruner=optuna.pruners.MedianPruner(
                n_startup_trials=15,
                n_warmup_steps=5,
                interval_steps=1,
            )
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True
        )

        def objective(trial: optuna.Trial) -> float:
            """Define hyperparameter search space"""
            trial_config = copy.deepcopy(config)

            # Optimize configuration values
            trial_config.setdefault("dataset", {})
            trial_config.setdefault("model", {})
            trial_config.setdefault("training", {})
            trial_config.setdefault("flow_matching", {})

           # Model mamba3
            trial_config["model"]["d_model"] = trial.suggest_categorical("d_model", [128, 256])
            trial_config["model"]["n_layers"] = trial.suggest_categorical("n_layers", [3,4])
            trial_config["model"]["d_state"] = trial.suggest_categorical("d_state", [32, 64])
            trial_config["model"]["expand"] = trial.suggest_categorical("expand", [2, 4])
            trial_config["model"]["headdim"] = 64  # Fixed at 64 to keep head calculations clean
            trial_config["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)

            trial_config["dataset"]["num_gated_bins"] = trial.suggest_int(
                "num_gated_bins", 2, 12)
            trial_config["dataset"]["window_size"] = trial.suggest_categorical(
                "window_size", [32, 64, 128] )
            trial_config["training"]["spectral_fft_weight"] = trial.suggest_float(
                "spectral_fft_weight", 1e-3,2e-1, log=True, )

            warmup_epochs = trial.suggest_categorical("warmup_epochs",[10, 20, 40, 60, 80],)
            fft_delay = trial.suggest_categorical("fft_delay",[10, 20, 30, 40])
            trial_config["training"]["warmup_epochs"] = warmup_epochs
            trial_config["training"]["fft_start_epoch"] = warmup_epochs + fft_delay
            # Training
            trial_config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)

            #Flow Matching
            trial_config["flow_matching"]["sigma_min"] = trial.suggest_float("sigma_min", 1e-6, 1e-2, log=True)

            try:
                return train_trial(trial, trial_config, device, use_optuna=args.use_optuna)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.empty_cache()
                logger.warning("Pruning trial due to runtime error: %s", e)

                raise optuna.TrialPruned()
        study.optimize(objective, n_trials=args.n_trials)

        logger.info("Optuna optimization completed successfully.")
        logger.info("Best Trial: %s", study.best_trial.number)
        logger.info("Best Value (FID): %s", study.best_value)
        logger.info("Best Params: %s", study.best_params)

if __name__ == "__main__":
    main()
