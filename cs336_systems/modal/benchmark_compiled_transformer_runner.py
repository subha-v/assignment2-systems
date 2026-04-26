import json

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

SIZE_NAMES = ["small", "medium", "large", "xl", "10B"]
MODES = ["forward", "forward_backward", "full"]
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
WARMUP_STEPS = 5
NUM_STEPS = 10


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=1800, max_containers=3)
def run_benchmark(size_name, mode, use_compile):
    import timeit
    import torch
    from cs336_basics.model import BasicsTransformerLM

    cfg = MODEL_CONFIGS[size_name]
    device = torch.device("cuda")
    print(f"running size={size_name} mode={mode} compile={use_compile} on {torch.cuda.get_device_name(0)}")

    try:
        model = BasicsTransformerLM(
            vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH,
            d_model=cfg["d_model"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=10_000.0,
        ).to(device)

        if use_compile is True:
            print(f"compiling model for size={size_name}")
            model = torch.compile(model)

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

        for _ in range(WARMUP_STEPS):
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
        sqDevs = []
        for t in timings:
            sqDevs.append((t - meanTime) ** 2)
        stdTime = (sum(sqDevs) / len(timings)) ** 0.5

        print(f"size={size_name} mode={mode} compile={use_compile}: {meanTime:.4f} +/- {stdTime:.4f} s")

        result = {}
        result["size"] = size_name
        result["mode"] = mode
        result["use_compile"] = use_compile
        result["mean"] = meanTime
        result["std"] = stdTime
        result["status"] = "ok"
        return result

    except torch.cuda.OutOfMemoryError:
        print(f"size={size_name} mode={mode} compile={use_compile}: OOM")
        result = {}
        result["size"] = size_name
        result["mode"] = mode
        result["use_compile"] = use_compile
        result["mean"] = None
        result["std"] = None
        result["status"] = "OOM"
        return result


def lookup_result(results, size_name, mode, use_compile):
    for r in results:
        if r["size"] == size_name and r["mode"] == mode and r["use_compile"] == use_compile:
            return r
    return None


def make_compare_table(results):
    text = ""
    text = text + "\\begin{table}\n"
    text = text + "\\centering\n"
    text = text + "\\begin{tabular}{llrr}\n"
    text = text + "\\toprule\n"
    text = text + "size & mode & vanilla (s) & compiled (s) \\\\\n"
    text = text + "\\midrule\n"

    for sizeName in SIZE_NAMES:
        for mode in MODES:
            vanillaResult = lookup_result(results, sizeName, mode, False)
            compiledResult = lookup_result(results, sizeName, mode, True)

            if vanillaResult is not None and vanillaResult["status"] == "ok":
                vanillaCell = f"{vanillaResult['mean']:.4f} $\\pm$ {vanillaResult['std']:.4f}"
            else:
                vanillaCell = "OOM"

            if compiledResult is not None and compiledResult["status"] == "ok":
                compiledCell = f"{compiledResult['mean']:.4f} $\\pm$ {compiledResult['std']:.4f}"
            else:
                compiledCell = "OOM"

            text = text + f"{sizeName} & {mode} & {vanillaCell} & {compiledCell} \\\\\n"

    text = text + "\\bottomrule\n"
    text = text + "\\end{tabular}\n"
    text = text + "\\caption{Per-step time (mean $\\pm$ std over 10 steps): vanilla Transformer vs torch.compile, ctx=512, batch=4.}\n"
    text = text + "\\label{tab:transformer_compile_compare}\n"
    text = text + "\\end{table}\n"
    return text


@app.local_entrypoint()
def main():
    jobs = []
    for sizeName in SIZE_NAMES:
        for mode in MODES:
            jobs.append((sizeName, mode, False))
    for sizeName in SIZE_NAMES:
        for mode in MODES:
            jobs.append((sizeName, mode, True))

    allResults = []
    for r in run_benchmark.starmap(jobs):
        allResults.append(r)

    print("")
    print("results:")
    for r in allResults:
        if r["status"] == "ok":
            print(f"size={r['size']} mode={r['mode']} compile={r['use_compile']}: {r['mean']:.4f} +/- {r['std']:.4f} s")
        else:
            print(f"size={r['size']} mode={r['mode']} compile={r['use_compile']}: {r['status']}")

    tableLatex = make_compare_table(allResults)

    with open("transformer_compile_results.json", "w") as f:
        json.dump(allResults, f, indent=2)
    with open("transformer_compile_compare.tex", "w") as f:
        f.write(tableLatex)

    print("saved results to transformer_compile_results.json and transformer_compile_compare.tex")
