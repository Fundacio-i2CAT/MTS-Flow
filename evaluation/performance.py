import pandas as pd
import torch
from thop import profile

from utils.config import load_config
from architecture.model import build_model


CONFIG_PATH = "configs/ett_336_best.yaml"

SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 6000, 9000, 18000]

N_STEPS = 100


@torch.no_grad()
def generate_sample(model, shape, device, n_steps=100):
    """
    Generate one sample using the same Euler integration
    procedure used during validation.
    """
    B, L, F = shape
    B=1

    x = torch.randn(
        B, L, F,
        device=device
    )

    for step in range(n_steps):
        t = torch.ones(
            (B,),
            device=device
        ) * (step * 0.01)

        v = model(x, t)

        x = x + v * 0.01

    return x


def benchmark_model(
    model,
    sample_input,
    warmup_runs=10,
    test_runs=100,
    n_steps=100
):
    device = torch.device("cuda")

    model = model.to(device)
    sample_input = sample_input.to(device)
    model.eval()

    B = sample_input.shape[0]

    # FLOPs: one forward pass
    sample_t = torch.rand(
        B,
        device=device
    )

    macs, params = profile(
        model,
        inputs=(sample_input, sample_t),
        verbose=False
    )

    forward_mflops = (2 * macs) / 1e6
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 100 Euler steps = 100 model evaluations
    generation_mflops = forward_mflops * n_steps

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            generate_sample(
                model,
                sample_input.shape,
                device,
                n_steps=n_steps
            )

    torch.cuda.synchronize()

    # VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = generate_sample(
            model,
            sample_input.shape,
            device,
            n_steps=n_steps
        )

    torch.cuda.synchronize()

    peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    peak_vram_mb = peak_vram_bytes / (1024 ** 2)

    # Generation time
    latencies = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():

        for _ in range(test_runs):

            start_event.record()

            _ = generate_sample(
                model,
                sample_input.shape,
                device,
                n_steps=n_steps
            )

            end_event.record()

            torch.cuda.synchronize()

            latencies.append(
                start_event.elapsed_time(end_event)
            )

    avg_time_ms = sum(latencies) / len(latencies)

    return {
        "mflops": generation_mflops,
        "vram_mb": peak_vram_mb,
        "time_ms": avg_time_ms,
        "params_m": params / 1e6
    }


def main():

    config = load_config(CONFIG_PATH)

    results = []

    num_features = config.get(
        "dataset", {}
    ).get(
        "num_features",
        10
    )

    for seq_len in SEQ_LENGTHS:

        print("\n" + "=" * 60)
        print(f"Sequence length: {seq_len}")
        print(f"Flow matching steps: {N_STEPS}")
        print("=" * 60)

        config.setdefault(
            "dataset",
            {}
        )["sequence_length"] = seq_len
        
        model = build_model(config)

        # One sample: (1, sequence_length, num_features)
        sample_input = torch.randn(
            1,
            seq_len,
            num_features
        )

        metrics = benchmark_model(
            model=model,
            sample_input=sample_input,
            warmup_runs=10,
            test_runs=100,
            n_steps=N_STEPS
        )

        result = {
            "Seq_Len": seq_len,
            "Flow_Steps": N_STEPS,
            "MFLOPS": round(
                metrics["mflops"],
                2
            ),
            "Inf_Time_ms": round(
                metrics["time_ms"],
                4
            ),
            "Max_VRAM_MB": round(
                metrics["vram_mb"],
                2
            ),
            "Params_M": round(
                metrics["params_m"],
                4
            )
        }

        results.append(result)

        print(f"MFLOPs:       {result['MFLOPS']}")
        print(f"Inference:    {result['Inf_Time_ms']} ms")
        print(f"Peak VRAM:    {result['Max_VRAM_MB']} MB")
        print(f"Parameters:   {result['Params_M']} M")

        pd.DataFrame(results).to_csv(
            "benchmark_results.csv",
            index=False
        )

        del model
        del sample_input

        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        pd.DataFrame(results).to_string(index=False)
    )

    print("\nSaved to: benchmark_results.csv")


if __name__ == "__main__":
    main()