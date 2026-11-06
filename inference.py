import argparse
import torch
import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.config import load_config
from architecture.model import build_model
from architecture.flow_matching import FlowMatcher

logger = get_logger("inference")


@torch.no_grad()
def euler_solver(model: torch.nn.Module, x: torch.Tensor, steps: int, device: torch.device) -> torch.Tensor:
    """
    Solves the ODE using the simple Euler method.
    dx = v_theta(x, t) dt
    """
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.ones((x.shape[0],), device=device) * (step * dt)
        v = model(x, t)
        x = x + v * dt
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--ode-steps", type=int, default=100, help="Number of ODE steps (NFE)")
    parser.add_argument("--output", type=str, default="generated_samples.npy", help="Output file path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup model
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully.")

    matcher = FlowMatcher()

    seq_len = config.get("dataset", {}).get("sequence_length", 512)
    num_features = config.get("dataset", {}).get("num_features", 10)

    shape = (args.samples, seq_len, num_features)
    x_0 = matcher.sample_noise(shape, device)

    logger.info(f"Generating {args.samples} samples with {args.ode_steps} ODE steps...")

    # Solve ODE
    x_generated = euler_solver(model, x_0, args.ode_steps, device)

    # Save output
    x_out = x_generated.cpu().numpy()

    if args.output.endswith(".npy"):
        np.save(args.output, x_out)
    elif args.output.endswith(".parquet"):
        # For saving to parquet, we typically need 2D. 
        # We flatten the batch and sequence dimension, and add IDs
        b_idx = np.repeat(np.arange(args.samples), seq_len)
        t_idx = np.tile(np.arange(seq_len), args.samples)
        df = pd.DataFrame(x_out.reshape(-1, num_features), columns=[f"feat_{i}" for i in range(num_features)])
        df.insert(0, "time", t_idx)
        df.insert(0, "batch_id", b_idx)
        df.to_parquet(args.output)
    else:
        logger.error("Output format not supported. Use .npy or .parquet")
        return

    logger.info(f"Saved generated samples to {args.output}")


if __name__ == "__main__":
    main()
