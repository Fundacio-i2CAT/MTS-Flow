"""Predictive and Discriminative evaluation"""
import os, sys
print("CWD:", os.getcwd())
print("PYTHON:", sys.executable)
print("PATH:", sys.path)
import utils
print("UTILS PATH:", list(utils.__path__))
import utils.config

import os
import mlflow
import torch
import numpy as np
import h5py
from  evaluation.predictive_discriminative import discriminative as disc
from  evaluation.predictive_discriminative import predictive as pred
from  evaluation.predictive_discriminative.load_synthetic import SyntheticGenerator
import utils.mlflow_utils as ut
 
from data.dataset import TimeSeriesDataset


def discriminative_predictive_evaluation(n_repetitions=2, tracking_uri='http://localhost:5000',
                                         run_id="f1bc0bef91394a86af291387a102a53a", epoch=-1, run_name="test"):
    """
    Gets config from artifacts in mlflow, reads real data from dataset in config,
    generates synthetic data and computes predictive and discriminative metrics.
    """
    syn_generator = SyntheticGenerator(run_id=run_id, tracking_uri=tracking_uri, epoch=epoch)
    config = syn_generator.config

    ut.setup_mlflow({
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": "discriminative_predictive_groundtruth"
        }
    })

    model_type = config.get("type") or config.get("model", {}).get("architecture", "unknown")
    dataset_path = config.get("dataset", {}).get("data_path", "")
    dataset_name = os.path.basename(dataset_path) if dataset_path else "unknown"

    metrics = {
        "p_mse": [],
        "d_score": [],
        "d_acc": [],
    }

    seq_len = config.get("data", {}).get("seq_len") or config.get("dataset", {}).get("sequence_length")

    with mlflow.start_run(run_name=run_name):
        ut.log_params({
            "model": model_type,
            "dataset": dataset_name,
            "eval_run_id": run_id,
            "eval_epoch": epoch
        })

        for rep in range(n_repetitions):
            with mlflow.start_run(run_name=f"rep_{rep+1}", nested=True):
                with h5py.File(config['dataset']['data_path'], 'r') as f:
                    train_data = f['train'][:]
                    validation_data = f['val'][:]
                    test_data = f['test'][:]

                if seq_len is not None:
                    train_data = train_data[:, :seq_len, :]
                    validation_data = validation_data[:, :seq_len, :]
                    test_data = test_data[:, :seq_len, :]

                train_set = TimeSeriesDataset(
                    train_data,
                    normalization=config["dataset"]["normalization"],
                    stats=None,
                    window_size=64,
                    num_gated_bins=0  # set to 0 avoids all the FT
                )
                train_stats = train_set.stats
                validation_set = TimeSeriesDataset(
                    validation_data,
                    normalization=config["dataset"]["normalization"],
                    stats=train_stats,
                    window_size=64,
                    num_gated_bins=0  # set to 0 avoids all the FT
                )
                test_set = TimeSeriesDataset(
                    test_data,
                    normalization=config["dataset"]["normalization"],
                    stats=train_stats,
                    window_size=64,
                    num_gated_bins=0  # set to 0 avoids all the FT
                )

                validation = validation_set.joint_data.cpu().numpy()
                test = test_set.joint_data.cpu().numpy()
                
                seq_len = validation_set.joint_data.shape[1]
                num_features = validation_set.joint_data.shape[2]

                syn_samples = train_set.joint_data.cpu().numpy()
                #syn_samples = syn_generator.generate_samples(len(validation_set), seq_len=seq_len, num_features=num_features)
                #extra_samples = syn_generator.generate_samples(len(test_set), seq_len=seq_len, num_features=num_features)
                samples_disc = train_set.joint_data.cpu().numpy() # np.concatenate((syn_samples, extra_samples), axis=0)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                mse = pred.predictive_score_metrics(test, syn_samples, device, num_features=num_features)
                acc, discriminative_score = disc.discriminative_score_metrics(
                    validation, test, samples_disc, device, num_features=num_features
                )

                ut.log_metrics({
                    "p_mse": mse,
                    "d_score": discriminative_score,
                    "d_acc": acc
                }, step=rep)

                metrics['p_mse'].append(mse)
                metrics['d_score'].append(discriminative_score)
                metrics['d_acc'].append(acc)

                print("Discriminative accuracy: ", acc, "\n Discriminative score: ", discriminative_score)

        summary_metrics = {}
        for key, value in metrics.items():
            summary_metrics[key] = float(np.mean(value))
            summary_metrics[f"{key}_std"] = float(np.std(value))

        ut.log_metrics(summary_metrics)


if __name__ == "__main__":

    b = [
        ("2164c92dcd724057a408e9d6d7b57429", "ideko_18300_pred_disc", 950),
    ]
    for i, run_name, e in b:
        discriminative_predictive_evaluation(n_repetitions=10, run_id=i, epoch=e, run_name=run_name)

