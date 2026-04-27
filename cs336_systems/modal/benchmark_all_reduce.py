### PROBLEM distributed_communication_single_node

import json
import os

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


WORLD_SIZES = [2, 4, 6]
# in mb
TENSOR_SIZES_MB = [1, 10, 100, 1024]
WARMUP_ITERS = 5
TIMED_ITERS = 20


# this is the function each spawned process runs
def worker(rank, world_size, tensor_bytes, result_path):
    import time
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    n_elements = tensor_bytes // 4
    data = torch.randn(n_elements, device=f"cuda:{rank}", dtype=torch.float32)

    for i in range(WARMUP_ITERS):
        dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()

    timings = []
    for i in range(TIMED_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms

    avg_ms = sum(timings) / len(timings)

    gathered = [None] * world_size
    dist.all_gather_object(gathered, avg_ms)

    if rank == 0:
        with open(result_path, "w") as f:
            json.dump(gathered, f)

    dist.destroy_process_group()


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200:6", timeout=1800, max_containers=3)
def run_all_reduce_benchmark(world_size, tensor_bytes):
    import tempfile
    import torch
    import torch.multiprocessing as mp

    size_mb = tensor_bytes // (1024 * 1024)
    print(f"running world_size={world_size}, size={size_mb} MB")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")

# goes back to parent
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
        result_path = tf.name

    mp.spawn(
        fn=worker,
        args=(world_size, tensor_bytes, result_path),
        nprocs=world_size,
        join=True,
    )

    with open(result_path) as f:
        per_rank_ms = json.load(f)

    avg_ms = sum(per_rank_ms) / len(per_rank_ms)

    for i in range(len(per_rank_ms)):
        print(f"  rank {i}: {per_rank_ms[i]:.3f} ms")
    print(f"  avg across ranks: {avg_ms:.3f} ms")

    result = {}
    result["world_size"] = world_size
    result["tensor_bytes"] = tensor_bytes
    result["tensor_mb"] = tensor_bytes // (1024 * 1024)
    result["per_rank_ms"] = per_rank_ms
    result["avg_ms"] = avg_ms
    return result


def lookup_result(results, world_size, tensor_mb):
    for r in results:
        if r["world_size"] == world_size and r["tensor_mb"] == tensor_mb:
            return r
    return None


def make_table(results):
    text = ""
    text = text + "\\begin{table}\n"
    text = text + "\\centering\n"
    text = text + "\\begin{tabular}{l" + "r" * len(WORLD_SIZES) + "}\n"
    text = text + "\\toprule\n"
    headerCells = []
    for ws in WORLD_SIZES:
        headerCells.append(f"world\\_size = {ws}")
    text = text + "size (MB) & " + " & ".join(headerCells) + " \\\\\n"
    text = text + "\\midrule\n"

    for size_mb in TENSOR_SIZES_MB:
        cells = []
        for ws in WORLD_SIZES:
            r = lookup_result(results, ws, size_mb)
            if r is not None:
                cells.append(f"{r['avg_ms']:.3f}")
            else:
                cells.append("--")
        text = text + f"{size_mb} & " + " & ".join(cells) + " \\\\\n"

    text = text + "\\bottomrule\n"
    text = text + "\\end{tabular}\n"
    text = text + "\\caption{All-reduce latency (ms) on B200 with NCCL, float32 tensors, averaged over ranks and 20 iterations after 5 warmup steps.}\n"
    text = text + "\\label{tab:all_reduce_single_node}\n"
    text = text + "\\end{table}\n"
    return text


@app.local_entrypoint()
def main():
    allResults = []
    for ws in WORLD_SIZES:
        for size_mb in TENSOR_SIZES_MB:
            tensor_bytes = size_mb * 1024 * 1024
            r = run_all_reduce_benchmark.remote(ws, tensor_bytes)
            allResults.append(r)

    print("")
    print("results:")
    for r in allResults:
        print(f"world_size={r['world_size']}, size={r['tensor_mb']} MB: {r['avg_ms']:.3f} ms")

    tableLatex = make_table(allResults)
    print("")
    print("latex table:")
    print(tableLatex)

    with open("all_reduce_benchmark_results.json", "w") as f:
        json.dump(allResults, f, indent=2)
    with open("all_reduce_benchmark_table.tex", "w") as f:
        f.write(tableLatex)

    print("saved all_reduce_benchmark_results.json and all_reduce_benchmark_table.tex")
