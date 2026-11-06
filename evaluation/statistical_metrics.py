import torch
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.signal import welch
from sklearn.metrics.pairwise import cosine_similarity
from tsfeatures import tsfeatures
import warnings
from tqdm import tqdm
import time


def convert_to_tsfeatures_format(ts_array: np.ndarray) -> pd.DataFrame:
    """
    Convert a 2D array of shape (num_sequences, sequence_length)
    to a DataFrame in tsfeatures format: columns = ["unique_id", "ds", "y"]
    """
    num_seq, seq_len = ts_array.shape
    df = pd.DataFrame({
        "unique_id": np.repeat(np.arange(num_seq), seq_len),
        "ds": np.tile(pd.date_range(start='2023-01-01', periods=seq_len, freq='D'), num_seq),
        "y": ts_array.flatten()
    })
    return df

def normalize_for_jsd(seq: np.ndarray) -> np.ndarray:
    """For jensen shannon values must be non negative and normalized."""
    seq = seq - np.min(seq) if np.min(seq) < 0 else seq
    seq = seq + 1e-8
    return seq / np.sum(seq)

def average_fid_score(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute the Fréchet Inception Distance (FID) score."""
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])

    cov_sqrt, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = np.sum((mu_real - mu_fake)**2) + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return float(np.mean(fid))

def average_cosine_similarity(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute cosine similarity between feature representations."""
    similarities = cosine_similarity(fake_features, real_features)
    mean_similarities = np.mean(similarities, axis=1)
    return float(np.mean(mean_similarities))

def average_jensen_shannon_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute Jensen-Shannon distance between feature representations."""
    real_arr = np.array([normalize_for_jsd(r) for r in real_features])
    fake_arr = np.array([normalize_for_jsd(s) for s in fake_features])

    jsd_matrix = cdist(fake_arr, real_arr, metric='jensenshannon')
    distances = np.mean(jsd_matrix, axis=1)
    return float(np.mean(distances))


def average_psd_similarity(
    real: np.ndarray,
    fake: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
) -> float:
    """
    Compute average Power Spectral Density similarity between real and fake signals.
    
    real, fake shape: (B, L)
    """
    if nperseg is None:
        nperseg = min(256, real.shape[1])

    _, psd_real = welch(real, fs=fs, nperseg=nperseg, axis=-1)
    _, psd_fake = welch(fake, fs=fs, nperseg=nperseg, axis=-1)

    mean_real_psd = np.mean(psd_real, axis=0)
    mean_fake_psd = np.mean(psd_fake, axis=0)

    diff_norm = np.linalg.norm(mean_real_psd - mean_fake_psd)
    real_norm = np.linalg.norm(mean_real_psd)

    similarity = 1.0 - diff_norm / (real_norm + 1e-12)
    return float(similarity)


def average_log_psd_mae(
    real: np.ndarray,
    fake: np.ndarray,
    fs: float = 1.0,
    nperseg: int | None = None,
) -> float:
    """
    Compute Mean Absolute Error (MAE) between average log-PSDs of real and fake signals.
    
    real, fake shape: (B, L)
    """
    if nperseg is None:
        nperseg = min(256, real.shape[1])

    _, psd_real = welch(real, fs=fs, nperseg=nperseg, axis=-1)
    _, psd_fake = welch(fake, fs=fs, nperseg=nperseg, axis=-1)

    mean_real_psd = np.mean(psd_real, axis=0)
    mean_fake_psd = np.mean(psd_fake, axis=0)

    log_real = np.log(mean_real_psd + 1e-12)
    log_fake = np.log(mean_fake_psd + 1e-12)

    mae = np.mean(np.abs(log_real - log_fake))
    return float(mae)


class MetricRegistry:
    def __init__(self):
        pass

    def evaluate(self, real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
        """
        Evaluate temporal metrics using extracted tsfeatures on generated synthetic data.
        real, fake shape: (B, L, F)
        """
        real_np = real.detach().cpu().numpy()
        fake_np = fake.detach().cpu().numpy()
        fake_np = np.nan_to_num(fake_np, nan=0.0)

        _, _, F = real_np.shape

        avg_fid_list = []
        avg_cosine_list = []
        avg_jsd_list = []
        avg_psd_list = []
        avg_log_psd_mae_list = []

        timings = {
            "real_tsfeatures": [],
            "fake_tsfeatures": [],
            "fid": [],
            "cosine": [],
            "jsd": [],
            "psd": [],
            "log_psd_mae": []
        }

        # Disable tsfeatures warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(F):
                #"""
                real_feat_df = convert_to_tsfeatures_format(real_np[:, :, i])
                fake_feat_df = convert_to_tsfeatures_format(fake_np[:, :, i])

                # --- Real Features Extraction ---
                t0 = time.perf_counter()
                real_feat = pd.concat( # Using tsfeatures in this way is much faster
                    [
                        tsfeatures(real_feat_df[real_feat_df["unique_id"] == uid], threads=15)
                        for uid in tqdm(real_feat_df["unique_id"].unique(), desc="Real features")
                    ],
                    ignore_index=True
                ).fillna(0).drop(columns=["unique_id"])

                timings["real_tsfeatures"].append(time.perf_counter() - t0)

                # --- Fake Features Extraction ---
                t0 = time.perf_counter()
                fake_feat = pd.concat(
                                    [
                                        tsfeatures(fake_feat_df[fake_feat_df["unique_id"] == uid], threads=15)
                                        for uid in tqdm(fake_feat_df["unique_id"].unique(), desc="Fake features")
                                    ],
                                    ignore_index=True
                                ).fillna(0).drop(columns=["unique_id"])
                timings["fake_tsfeatures"].append(time.perf_counter() - t0)

                # --- FID Score ---
                t0 = time.perf_counter()
                fid = average_fid_score(real_feat.values, fake_feat.values)
                timings["fid"].append(time.perf_counter() - t0)

                # --- Cosine Similarity ---
                t0 = time.perf_counter()
                cosine = average_cosine_similarity(real_feat.values, fake_feat.values)
                timings["cosine"].append(time.perf_counter() - t0)

                # --- Jensen-Shannon Distance ---
                t0 = time.perf_counter()
                jsd = average_jensen_shannon_distance(real_feat.values, fake_feat.values)
                timings["jsd"].append(time.perf_counter() - t0)

                # --- PSD Similarity ---
                t0 = time.perf_counter()
                psd_sim = average_psd_similarity(real_np[:, :, i], fake_np[:, :, i])
                timings["psd"].append(time.perf_counter() - t0)

                # --- Log-PSD MAE ---
                t0 = time.perf_counter()
                log_psd_mae = average_log_psd_mae(real_np[:, :, i], fake_np[:, :, i])
                timings["log_psd_mae"].append(time.perf_counter() - t0)

                avg_fid_list.append(fid)
                avg_cosine_list.append(cosine)
                avg_jsd_list.append(jsd)
                avg_psd_list.append(psd_sim)
                avg_log_psd_mae_list.append(log_psd_mae)

                # --- Final Sum and Print Summary ---
            print("\n" + "="*50)
            print("             EXECUTION TIMING PROFILE             ")
            print("="*50)
            for step, duration_list in timings.items():
                total_time = sum(duration_list)
                avg_time = total_time / len(duration_list) if duration_list else 0
                step_name = step.replace("_", " ").title()
                print(f"{step_name:<25} | Total: {total_time:7.3f}s | Avg/Iter: {avg_time:6.3f}s")
            print("="*50)
        print(avg_fid_list)
        return {
            "frechet_inception_distance": float(np.mean(avg_fid_list)),
            "cosine_similarity": float(np.mean(avg_cosine_list)),
            "jensen_shannon_divergence": float(np.mean(avg_jsd_list)),
            "psd_similarity": float(np.mean(avg_psd_list)),
            "log_psd_mae": float(np.mean(avg_log_psd_mae_list))
        }


def compute_metrics(real: np.ndarray | torch.Tensor, fake: np.ndarray | torch.Tensor) -> dict[str, float]:
    """
    Compute temporal and frequency metrics using MetricRegistry.
    Accepts numpy arrays or torch Tensors of shape (B, L, F).
    """
    if isinstance(real, np.ndarray):
        real = torch.from_numpy(real)
    if isinstance(fake, np.ndarray):
        fake = torch.from_numpy(fake)
    registry = MetricRegistry()
    return registry.evaluate(real, fake)

