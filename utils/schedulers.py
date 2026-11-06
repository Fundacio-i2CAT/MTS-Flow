from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


def build_scheduler(optimizer, config):
    training_cfg = config.get("training", {})

    scheduler_name = training_cfg.get("scheduler", "none").lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        epochs = training_cfg["epochs"]
        eta_min = float(training_cfg.get("min_lr"))
        warmup_epochs = int(training_cfg.get("warmup_epochs", 0))

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=eta_min,
        )

        if warmup_epochs <= 0:
            return cosine_scheduler

        warmup_start_factor = float(training_cfg.get("warmup_start_factor", 0.01))

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    raise ValueError(f"Unknown scheduler {scheduler_name}")