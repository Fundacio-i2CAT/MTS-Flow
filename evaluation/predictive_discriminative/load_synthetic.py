"""Class for inference and create samples
Made them to a Pytorch Dataset
"""
import re
import os
import gc
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from architecture.model import build_model
from utils.config import load_config


class SyntheticGenerator(Dataset):
    """ Synthetic data generation"""
    def __init__(self,
                 run_id='f1bc0bef91394a86af291387a102a53a',
                 tracking_uri='http://localhost:5007',
                 epoch=-1
                 ):

        self.run_id = run_id
        self.tracking_uri = tracking_uri

        # Load generator checkpoint from MLflow
        latest_checkpoint_path = self._get_latest_checkpoint_path(run_id, epoch)
        self.fm_ckp = self._load_checkpoint(run_id, latest_checkpoint_path)
        config_path = self._get_config(run_id)
        self.config = load_config(config_path)
        self.fm_net = build_model(self.config)

        # self.gen_net = self.gen_net(self.gen_ckp['model_config']['generator'])
        self.fm_net.load_state_dict(self.fm_ckp['model_state_dict'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fm_net.to(self.device)
        self.syn_sample = None

    def generate_samples(self, samples: int, seq_len: int, num_features: int):
        """Generate synthetic samples using Euler integration."""

        self.fm_net.eval()

        n_steps = 100
        dt = 1.0 / n_steps

        # (B, L, F)
        x = torch.randn(
            samples,
            seq_len,
            num_features,
            device=self.device
        )

        with torch.no_grad():
            for step in range(n_steps):
                t = torch.ones(
                    (samples,),
                    device=self.device
                ) * (step * dt)

                v = self.fm_net(x, t)
                x = x + v * dt

        self.syn_sample = x.cpu().numpy()

        return self.syn_sample

    def _get_config(self, run_id: str):
        mlflow.set_tracking_uri(self.tracking_uri)
        client = mlflow.tracking.MlflowClient()

        artifacts = client.list_artifacts(run_id, path="")
        config = [artifact.path for artifact in artifacts if artifact.path.endswith(".yaml")]
        config = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{config[0]}")
        return config

    def _get_latest_checkpoint_path(self, run_id: str, epoch):
        """Find the latest checkpoint file (highest-numbered) in the 'checkpoints/' folder."""
        if epoch == -1:
            mlflow.set_tracking_uri(self.tracking_uri)
            client = mlflow.tracking.MlflowClient()

            # List artifacts in the "checkpoints" folder
            artifacts = client.list_artifacts(run_id, path="checkpoints")
            checkpoint_files = [artifact.path for artifact in artifacts if artifact.path.endswith(".pt")]

            # Extract numbers from filenames like "235_checkpoint.pth"
            checkpoint_numbers = []
            pattern = re.compile(r"best_model_epoch_(\d+)\.pt")

            for file in checkpoint_files:
                match = pattern.match(file.split("/")[-1])
                if match:
                    checkpoint_numbers.append(int(match.group(1)))

            if not checkpoint_numbers:
                raise ValueError("No valid checkpoint files found in MLflow artifacts.")

            # Find the file with the highest number (latest epoch)
            latest_checkpoint = max(checkpoint_numbers)
            latest_checkpoint_file = f"checkpoints/best_model_epoch_{latest_checkpoint}.pt"

        else:
            latest_checkpoint_file = f"checkpoints/best_model_epoch_{epoch}.pt"
        return latest_checkpoint_file

    def _load_checkpoint(self, run_id: str, checkpoint_path: str):
        """Load the generator model checkpoint from a given MLflow artifact path."""
        mlflow.set_tracking_uri(self.tracking_uri)
        local_checkpoint_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{checkpoint_path}")
        checkpoint = torch.load(local_checkpoint_path, map_location=torch.device('cpu'))
        return checkpoint


def save_time_series_plot(array, plot_name, plot_save_path):
    """Save time series data plot"""
    n_features = array.shape[-1]
    seq_len = array.shape[-2]

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(array.squeeze(0), columns=[f'f{i+1}' for i in range(n_features)])
    col = df.columns
    df['time'] = np.arange(seq_len)

    _, ax = plt.subplots(figsize=(10, 6))
    for i in col:
        ax.plot(df['time'], df[i], label=i)

    ax.set_xlabel('Time')
    ax.set_ylabel('Feature Value')
    ax.set_title('Synthetic Time Series Sample')
    plt.tight_layout()
    os.makedirs(plot_save_path, exist_ok=True)

    # Save the plot
    plot_file_path = os.path.join(plot_save_path, f"{plot_name}.png")
    plt.savefig(plot_file_path, bbox_inches='tight')

    # Clean up memory
    plt.clf()
    plt.close('all')
    gc.collect()

