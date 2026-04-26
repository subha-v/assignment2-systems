## PROBLEM pytorch_attention

import json

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384, 32768, 65536, 131072]         


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=7200, max_containers=3)
def run_attention_benchmark(d_model, seq_len):
    import torch
    from cs336_systems.benchmark_attention import time_one_config

    print(f"running d_model={d_model} seq_len={seq_len} on {torch.cuda.get_device_name(0)}")
    result = time_one_config(d_model, seq_len)
    if result["status"] == "ok":
        print(f"d_model={d_model} seq_len={seq_len}: forward={result['forward_ms']:.3f} ms, backward={result['backward_ms']:.3f} ms, mem={result['memory_before_backward_mib']:.1f} MiB")
    else:
        print(f"d_model={d_model} seq_len={seq_len}: {result['status']}")
    return result


def make_latex_table(results, value_key, value_format, caption, label):
    text = ""
    text = text + "\\begin{table}\n"
    text = text + "\\centering\n"

    colSpec = "l"
    for s in SEQ_LEN_VALUES:
        colSpec = colSpec + "c"
    text = text + "\\begin{tabular}{" + colSpec + "}\n"
    text = text + "\\toprule\n"

    headerRow = "d\\_model"
    for s in SEQ_LEN_VALUES:
        headerRow = headerRow + " & " + str(s)
    headerRow = headerRow + " \\\\\n"
    text = text + headerRow
    text = text + "\\midrule\n"

    for d in D_MODEL_VALUES:
        rowText = str(d)
        for s in SEQ_LEN_VALUES:
            cell = "OOM"
            for r in results:
                if r["d_model"] == d and r["seq_len"] == s:
                    if r["status"] == "ok":
                        cell = value_format.format(r[value_key])
                    break
            rowText = rowText + " & " + cell
        rowText = rowText + " \\\\\n"
        text = text + rowText

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
            r = run_attention_benchmark.remote(d, s)
            allResults.append(r)

    print("")
    print("results:")
    for r in allResults:
        if r["status"] == "ok":
            print(f"d_model={r['d_model']} seq_len={r['seq_len']}: forward={r['forward_ms']:.3f} ms, backward={r['backward_ms']:.3f} ms, mem={r['memory_before_backward_mib']:.1f} MiB")
        else:
            print(f"d_model={r['d_model']} seq_len={r['seq_len']}: {r['status']}")

    forwardLatex = make_latex_table(allResults, "forward_ms", "{:.3f}",
                                    "Forward pass time per call (ms), batch size 8.",
                                    "tab:attn_forward_ms")
    backwardLatex = make_latex_table(allResults, "backward_ms", "{:.3f}",
                                     "Backward pass time per call (ms), batch size 8.",
                                     "tab:attn_backward_ms")
    memoryLatex = make_latex_table(allResults, "memory_before_backward_mib", "{:.1f}",
                                   "Memory in use before the backward pass (MiB), batch size 8.",
                                   "tab:attn_memory_mib")

    with open("attention_benchmark_results.json", "w") as f:
        json.dump(allResults, f, indent=2)
    with open("attention_table_forward.tex", "w") as f:
        f.write(forwardLatex)
    with open("attention_table_backward.tex", "w") as f:
        f.write(backwardLatex)
    with open("attention_table_memory.tex", "w") as f:
        f.write(memoryLatex)

    print("saved results to attention_benchmark_results.json and 3 .tex files")
