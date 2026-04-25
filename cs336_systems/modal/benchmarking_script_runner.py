import json
import modal
import pandas as pd

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

MODES = ["forward", "forward_backward", "full"]
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
NUM_WARMUP = 5
NUM_STEPS = 10


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=1800, max_containers=3)
def run_benchmark(size_name: str, mode: str) -> dict:
    import timeit
    import torch
    from cs336_basics.model import BasicsTransformerLM

    cfg = MODEL_CONFIGS[size_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{size_name} | {mode}] device={device}, gpu={torch.cuda.get_device_name(0)}")

    try:
        model = BasicsTransformerLM(
            vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH,
            d_model=cfg["d_model"], num_layers=cfg["num_layers"],
            num_heads =cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=10_000.0,
        ).to(device)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if mode == "forward":
            def step_fn():
                model(x)
        elif mode == "forward_backward":
            def step_fn():
                model.zero_grad()
                y = model(x)
                lossOut = y.sum()
                lossOut.backward()
        elif mode == "full":
            def step_fn():
                optimizer.zero_grad()
                y = model(x)
                lossOut = y.sum()
                lossOut.backward()
                optimizer.step()

        for _ in range(NUM_WARMUP):
            step_fn()
            torch.cuda.synchronize()

        timings = []
        for _ in range(NUM_STEPS):
            torch.cuda.synchronize()
            t0 = timeit.default_timer()
            step_fn()
            torch.cuda.synchronize()
            t1 = timeit.default_timer()
            timings.append(t1 - t0)

        meanTime = sum(timings) / len(timings)
        sqDevs = [(t - meanTime) ** 2 for t in timings]
        stdTime = (sum(sqDevs) / len(timings)) ** 0.5

        print(f"[{size_name} | {mode}] {meanTime:.4f} +/- {stdTime:.4f} s")
        return {
            "size": size_name, "mode": mode,
            "mean": meanTime, "std": stdTime,
            "times": timings, "status": "ok",
        }

    except torch.cuda.OutOfMemoryError:
        print(f"[{size_name} | {mode}] OOM!")
        return {"size": size_name, "mode": mode, "mean": None, "std": None, "times": [], "status": "OOM"}
    except Exception as e:
        print(f"[{size_name} | {mode}] error: {e}")
        return {"size": size_name, "mode": mode, "mean": None, "std": None, "times": [], "status": f"error: {e}"}


def build_latex_table(results):
    sizeOrder = list(MODEL_CONFIGS.keys())
    rows = []
    for sizeName in sizeOrder:
        row = {"Size": sizeName}
        for mode in MODES:
            matching = [r for r in results if r["size"] == sizeName and r["mode"] == mode]
            if len(matching) == 0:
                row[mode] = "N/A"
                continue
            r = matching[0]
            if r["status"] != "ok":
                row[mode] = r["status"]
            else:
                row[mode] = f"{r['mean']:.4f} $\\pm$ {r['std']:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "forward": "Forward (s)",
        "forward_backward": "Fwd+Bwd (s)",
        "full": "Fwd+Bwd+Opt (s)",
    })

    colFmt = "l" + "c" * (len(df.columns) - 1)
    latexStr = df.to_latex(
        index=False, escape=False, column_format=colFmt,
        caption="End-to-end benchmarking results (mean $\\pm$ std over 10 steps, 5 warmup).",
        label="tab:benchmarking",
    )
    return latexStr


def print_summary(results):
    sizeOrder = list(MODEL_CONFIGS.keys())
    sortedResults = sorted(results, key=lambda r: (sizeOrder.index(r["size"]), MODES.index(r["mode"])))

    print("\n" + "=" * 90)
    print(f"{'Size':<10} {'Mode':<20} {'Mean (s)':<15} {'Std (s)':<15} {'Status'}")
    print("-" * 90)
    for r in sortedResults:
        if r["status"] == "ok":
            print(f"{r['size']:<10} {r['mode']:<20} {r['mean']:<15.4f} {r['std']:<15.4f} {r['status']}")
        else:
            print(f"{r['size']:<10} {r['mode']:<20} {'N/A':<15} {'N/A':<15} {r['status']}")
    print("=" * 90)


@app.local_entrypoint()
def main():
    jobs = []
    for sizeName in MODEL_CONFIGS:
        for mode in MODES:
            jobs.append((sizeName, mode))

    allResults = []
    for result in run_benchmark.starmap(jobs):
        allResults.append(result)

    print_summary(allResults)

    latexTable = build_latex_table(allResults)
    print("\nLaTeX Table:\n")
    print(latexTable)

    with open("benchmark_results.json", "w") as f:
        json.dump(allResults, f, indent=2)

    with open("benchmark_table.tex", "w") as f:
        f.write(latexTable)

    print("Saved benchmark_results.json and benchmark_table.tex")