import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def analyze_spectral_features(
    dataloader,
    output_dir: str,
    num_features: int,
    num_bins: int,
    max_samples: int = 50000,
):
    """
    Analyze the distribution of every spectral feature.

    Expected feature ordering:
        feature0_raw,
        feature0_bin0,
        ...
        feature0_binN,
        feature1_raw,
        feature1_bin0,
        ...

    Saves:
        feature_0_bin_00_hist.png
        feature_0_bin_01_hist.png
        ...
        summary.txt
    """

    os.makedirs(output_dir, exist_ok=True)

    stride = 1 + num_bins

    # Gather data
    collected = []

    with torch.no_grad():
        for batch in dataloader:

            if isinstance(batch, dict):
                x = batch["x"]
            else:
                x = batch

            collected.append(x.cpu())

            n = sum(b.shape[0] for b in collected)
            if n >= max_samples:
                break

    x = torch.cat(collected, dim=0)

    # flatten samples and time
    x = x.reshape(-1, x.shape[-1]).numpy()

    summary = []

    summary.append("===== Spectral Feature Statistics =====\n")

    for feature in range(num_features):

        for bin_idx in range(num_bins):

            col = feature * stride + 1 + bin_idx

            values = x[:, col]

            stats = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "p99": np.percentile(values, 99),
                "zeros": np.mean(values == 0),
            }

            summary.append(
                f"Feature {feature} Bin {bin_idx}\n"
                f"  min      : {stats['min']:.6f}\n"
                f"  max      : {stats['max']:.6f}\n"
                f"  mean     : {stats['mean']:.6f}\n"
                f"  std      : {stats['std']:.6f}\n"
                f"  median   : {stats['median']:.6f}\n"
                f"  p99      : {stats['p99']:.6f}\n"
                f"  zero %   : {100*stats['zeros']:.2f}\n"
            )

            fig, ax = plt.subplots(figsize=(7,5))

            ax.hist(values, bins=100)

            ax.set_title(
                f"Feature {feature} - FFT Bin {bin_idx}\n"
                f"mean={stats['mean']:.3f} "
                f"std={stats['std']:.3f}"
            )

            ax.set_xlabel("Normalized value")
            ax.set_ylabel("Count")

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    output_dir,
                    f"feature_{feature}_bin_{bin_idx:02d}_hist.png"
                )
            )

            plt.close(fig)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary))

    print(f"Saved spectral analysis to {output_dir}")