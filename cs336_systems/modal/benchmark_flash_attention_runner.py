## PROBLEM flash_benchmarking

import json

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


BATCH_SIZE = 1
SEQ_LEN_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODEL_VALUES = [16, 32, 64, 128]
DTYPE_NAMES = ["bfloat16", "float32"]
KINDS = ["triton", "pytorch"]
WARMUP = 10
REP = 50


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=3600, max_containers=3)
def run_flash_benchmark(seq_len, d_model, dtype_name, kind):
    import torch
    import triton.testing

    from cs336_basics.model import scaled_dot_product_attention
    from cs336_systems.flash_attention.triton_flash_attention import FlashAttentionTriton

    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"running kind={kind} dtype={dtype_name} d_model={d_model} seq_len={seq_len} on {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda")

    try:
        Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

        if kind == "triton":
            def fwd():
                return FlashAttentionTriton.apply(Q, K, V, True)
        else:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            def fwd():
                return scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                
        out = fwd()
        dout = torch.randn_like(out)
        out.backward(dout, retain_graph=True)
        Q.grad = None
        K.grad = None
        V.grad = None
        torch.cuda.synchronize()

        forward_ms = triton.testing.do_bench(fwd, warmup=WARMUP, rep=REP)

        out = fwd()

        def bwd():
            Q.grad = None
            K.grad = None
            V.grad = None
            out.backward(dout, retain_graph=True)

        backward_ms = triton.testing.do_bench(bwd, warmup=WARMUP, rep=REP)

        def fwd_bwd():
            Q.grad = None
            K.grad = None
            V.grad = None
            o = fwd()
            o.backward(dout)

        end_to_end_ms = triton.testing.do_bench(fwd_bwd, warmup=WARMUP, rep=REP)

        result = {
            "seq_len": seq_len,
            "d_model": d_model,
            "dtype": dtype_name,
            "kind": kind,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "end_to_end_ms": end_to_end_ms,
            "status": "ok",
        }
        print(f"  forward: {forward_ms:.3f} ms, backward: {backward_ms:.3f} ms, e2e: {end_to_end_ms:.3f} ms")
        return result

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        torch.cuda.empty_cache()
        print(f"  err ({type(e).__name__}): {e}")
        return {
            "seq_len": seq_len,
            "d_model": d_model,
            "dtype": dtype_name,
            "kind": kind,
            "forward_ms": None,
            "backward_ms": None,
            "end_to_end_ms": None,
            "status": f"err ({type(e).__name__})",
        }


def lookup_result(results, seq_len, d_model, dtype_name, kind):
    for r in results:
        if r["seq_len"] == seq_len and r["d_model"] == d_model and r["dtype"] == dtype_name and r["kind"] == kind:
            return r
    return None


def make_compare_table(results, value_key, value_format, dtype_name, caption, label):
    text = ""
    text = text + "\\begin{table}\n"
    text = text + "\\centering\n"
    text = text + "\\begin{tabular}{llrr}\n"
    text = text + "\\toprule\n"
    text = text + "d\\_model & seq\\_len & triton & pytorch \\\\\n"
    text = text + "\\midrule\n"

    for d in D_MODEL_VALUES:
        for s in SEQ_LEN_VALUES:
            tritonResult = lookup_result(results, s, d, dtype_name, "triton")
            pytorchResult = lookup_result(results, s, d, dtype_name, "pytorch")

            if tritonResult is None:
                tritonCell = "missing"
            elif tritonResult["status"] == "ok":
                tritonCell = value_format.format(tritonResult[value_key])
            else:
                tritonCell = tritonResult["status"]

            if pytorchResult is None:
                pytorchCell = "missing"
            elif pytorchResult["status"] == "ok":
                pytorchCell = value_format.format(pytorchResult[value_key])
            else:
                pytorchCell = pytorchResult["status"]

            text = text + f"{d} & {s} & {tritonCell} & {pytorchCell} \\\\\n"

    text = text + "\\bottomrule\n"
    text = text + "\\end{tabular}\n"
    text = text + "\\caption{" + caption + "}\n"
    text = text + "\\label{" + label + "}\n"
    text = text + "\\end{table}\n"
    return text


@app.local_entrypoint()
def main():
    jobs = []
    for dtype_name in DTYPE_NAMES:
        for d in D_MODEL_VALUES:
            for s in SEQ_LEN_VALUES:
                for kind in KINDS:
                    jobs.append((s, d, dtype_name, kind))

    allResults = []
    for r in run_flash_benchmark.starmap(jobs):
        allResults.append(r)

    print("")
    print("results:")
    for r in allResults:
        if r["status"] == "ok":
            print(f"dtype={r['dtype']} d_model={r['d_model']} seq_len={r['seq_len']} kind={r['kind']}: forward={r['forward_ms']:.3f} ms, backward={r['backward_ms']:.3f} ms, e2e={r['end_to_end_ms']:.3f} ms")
        else:
            print(f"dtype={r['dtype']} d_model={r['d_model']} seq_len={r['seq_len']} kind={r['kind']}: {r['status']}")

    allTables = []
    for dtype_name in DTYPE_NAMES:
        forwardLatex = make_compare_table(
            allResults, "forward_ms", "{:.3f}", dtype_name,
            f"Forward pass time per call (ms): triton vs pytorch, batch size 1, causal, dtype={dtype_name}.",
            f"tab:flash_forward_compare_{dtype_name}",
        )
        backwardLatex = make_compare_table(
            allResults, "backward_ms", "{:.3f}", dtype_name,
            f"Backward pass time per call (ms): triton vs pytorch, batch size 1, causal, dtype={dtype_name}.",
            f"tab:flash_backward_compare_{dtype_name}",
        )
        e2eLatex = make_compare_table(
            allResults, "end_to_end_ms", "{:.3f}", dtype_name,
            f"End-to-end forward+backward time per call (ms): triton vs pytorch, batch size 1, causal, dtype={dtype_name}.",
            f"tab:flash_e2e_compare_{dtype_name}",
        )
        allTables.append((dtype_name, "forward", forwardLatex))
        allTables.append((dtype_name, "backward", backwardLatex))
        allTables.append((dtype_name, "e2e", e2eLatex))

    with open("flash_attention_benchmark_results.json", "w") as f:
        json.dump(allResults, f, indent=2)
    for dtype_name, metric, latex in allTables:
        outPath = f"flash_attention_table_{metric}_{dtype_name}.tex"
        with open(outPath, "w") as f:
            f.write(latex)

    print(f"saved results to flash_attention_benchmark_results.json and {len(allTables)} .tex files")
