import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.mlflow_utils import log_artifact

def save_time_series_plots(real_data: np.ndarray, fake_data: np.ndarray, epoch: int, max_plots: int = 3):
    """
    Generate plots of real vs synthetic data and save to MLFlow.
    max_plots defines the number of samples to plot.
    real_data, fake_data: arrays of shape (Batch, SeqLen, Features)
    """
    num_samples = min(max_plots, fake_data.shape[0])
    
    if num_samples == 0:
        return
        
    indices = np.linspace(0, fake_data.shape[0] - 1, num_samples, dtype=int)
    
    seq_len = fake_data.shape[1]
    num_features = fake_data.shape[2]
    
    df = pd.DataFrame({'time': np.arange(seq_len)})
    
    for i, idx in enumerate(indices):
        for feat in range(num_features):
            df[f'generated{feat}_{i}'] = fake_data[idx][:, feat]
            df[f'real{feat}_{i}'] = real_data[idx][:, feat]
            
    fig, axs = plt.subplots(num_samples, 2, figsize=(13, 3 * num_samples))
    fig.suptitle(f"Epoch {epoch}: Real vs Generated Data", fontsize=16)
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)
        
    for i in range(num_samples):
        for j, key in enumerate(['generated', 'real']):
            axs[i, j].set_xlabel('Time')
            axs[i, j].set_ylabel(key, color='blue')
            for feat in range(num_features):
                axs[i, j].plot(df['time'], df[f'{key}{feat}_{i}'])
            axs[i, j].tick_params(axis='y', labelcolor='blue')
            
    plt.tight_layout()
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, f"plot_epoch_{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
        
        log_artifact(plot_path, "plots")

def save_synthetic_data(fake_data: np.ndarray, epoch: int):
    """
    Save synthetic data locally and to MLFlow.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, f"syn_data_epoch_{epoch}.npy")
        np.save(data_path, fake_data)
        log_artifact(data_path, "syn_data")
