import json

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=3600, max_containers=3)
def run_attention_benchmark(d_model, seq_len, use_compile):
    import torch
    from cs336_systems.benchmark_attention import time_one_config

    print(f"running d_model={d_model} seq_len={seq_len} compile={use_compile} on {torch.cuda.get_device_name(0)}")
    result = time_one_config(d_model, seq_len, use_compile=use_compile)
    if result["status"] == "ok":
        print(f"d_model={d_model} seq_len={seq_len} compile={use_compile}: forward={result['forward_ms']:.3f} ms, backward={result['backward_ms']:.3f} ms, mem={result['memory_before_backward_mib']:.1f} MiB")
    else:
        print(f"d_model={d_model} seq_len={seq_len} compile={use_compile}: {result['status']}")
    return result


def lookup_result(results, d_model, seq_len, use_compile):
    for r in results:
        if r["d_model"] == d_model and r["seq_len"] == seq_len and r["use_compile"] == use_compile:
            return r
    return None


def make_compare_table(results, value_key, value_format, caption, label):
    text = ""
    text = text + "\\begin{table}\n"
    text = text + "\\centering\n"
    text = text + "\\begin{tabular}{llrr}\n"
    text = text + "\\toprule\n"
    text = text + "d\\_model & seq\\_len & vanilla & compiled \\\\\n"
    text = text + "\\midrule\n"

    for d in D_MODEL_VALUES:
        for s in SEQ_LEN_VALUES:
            vanillaResult = lookup_result(results, d, s, False)
            compiledResult = lookup_result(results, d, s, True)

            if vanillaResult is not None and vanillaResult["status"] == "ok":
                vanillaCell = value_format.format(vanillaResult[value_key])
            else:
                vanillaCell = "OOM"

            if compiledResult is not None and compiledResult["status"] == "ok":
                compiledCell = value_format.format(compiledResult[value_key])
            else:
                compiledCell = "OOM"

            text = text + f"{d} & {s} & {vanillaCell} & {compiledCell} \\\\\n"

    text = text + "\\bottomrule\n"
    text = text + "\\end{tabular}\n"
    text = text + "\\caption{" + caption + "}\n"
    text = text + "\\label{" + label + "}\n"
    text = text + "\\end{table}\n"
    return text


@app.local_entrypoint()
def main():
    allResults = []
    for d in D_MODEL_VALUES:
        for s in SEQ_LEN_VALUES:
            r = run_attention_benchmark.remote(d, s, False)
            allResults.append(r)
    for d in D_MODEL_VALUES:
        for s in SEQ_LEN_VALUES:
            r = run_attention_benchmark.remote(d, s, True)
            allResults.append(r)

    print("")
    print("results:")
    for r in allResults:
        if r["status"] == "ok":
            print(f"d_model={r['d_model']} seq_len={r['seq_len']} compile={r['use_compile']}: forward={r['forward_ms']:.3f} ms, backward={r['backward_ms']:.3f} ms, mem={r['memory_before_backward_mib']:.1f} MiB")
        else:
            print(f"d_model={r['d_model']} seq_len={r['seq_len']} compile={r['use_compile']}: {r['status']}")

    forwardLatex = make_compare_table(allResults, "forward_ms", "{:.3f}",
                                      "Forward pass time per call (ms): vanilla vs torch.compile, batch size 8.",
                                      "tab:attn_forward_compare")
    backwardLatex = make_compare_table(allResults, "backward_ms", "{:.3f}",
                                       "Backward pass time per call (ms): vanilla vs torch.compile, batch size 8.",
                                       "tab:attn_backward_compare")

    with open("attention_benchmark_results.json", "w") as f:
        json.dump(allResults, f, indent=2)
    with open("attention_table_forward_compare.tex", "w") as f:
        f.write(forwardLatex)
    with open("attention_table_backward_compare.tex", "w") as f:
        f.write(backwardLatex)

    print("saved results to attention_benchmark_results.json and 2 .tex files")
