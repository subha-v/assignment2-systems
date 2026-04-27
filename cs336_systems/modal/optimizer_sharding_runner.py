## PROBLEM optimizer_state_sharding_accounting
import json
import os

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


VOCAB_SIZE = 10000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
D_MODEL = 2560
D_FF = 10240
NUM_LAYERS = 32
NUM_HEADS = 32

WARMUP_ITERS = 3
TIMED_ITERS = 5


def worker(rank, world_size, use_sharding, result_path):
    import time
    import torch
    import torch.distributed as dist
    import torch.optim as optim
    import torch.nn.functional as F
    from cs336_basics.model import BasicsTransformerLM
    from cs336_systems.optimizer_sharding import OptimizerSharding

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"rank {rank}: building xl model on {device}, use_sharding={use_sharding}")
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=10000.0,
    ).to(device)

    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)

    torch.cuda.synchronize()
    peak_after_init = torch.cuda.max_memory_allocated(device)
    mem_after_init = torch.cuda.memory_allocated(device)

    if use_sharding:
        optimizer = OptimizerSharding(model.parameters(), optim.AdamW, lr=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    local_batch_size = BATCH_SIZE // world_size
    torch.manual_seed(41)
    all_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)
    all_y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)

    offset = rank * local_batch_size
    local_x = all_x[offset : offset + local_batch_size]
    local_y = all_y[offset : offset + local_batch_size]

    for i in range(WARMUP_ITERS):
        optimizer.zero_grad()
        logits = model(local_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), local_y.reshape(-1))
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size
        optimizer.step()
    torch.cuda.synchronize()

    step_times = []
    peak_before_step_list = []
    peak_after_step_list = []
    mem_before_step_list = []
    mem_after_step_list = []

    for i in range(TIMED_ITERS):
        optimizer.zero_grad()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()

        logits = model(local_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), local_y.reshape(-1))
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size

        torch.cuda.synchronize()
        peak_before = torch.cuda.max_memory_allocated(device)
        mem_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        optimizer.step()

        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated(device)
        mem_after = torch.cuda.memory_allocated(device)

        t1 = time.perf_counter()

        step_times.append((t1 - t0) * 1000.0)
        peak_before_step_list.append(peak_before)
        peak_after_step_list.append(peak_after)
        mem_before_step_list.append(mem_before)
        mem_after_step_list.append(mem_after)

    total_time = 0.0
    for t in step_times:
        total_time += t
    avg_step_ms = total_time / len(step_times)

    # use max across iterations  for stability
    peak_before_avg = max(peak_before_step_list)
    peak_after_avg = max(peak_after_step_list)
    mem_before_avg = sum(mem_before_step_list) / len(mem_before_step_list)
    mem_after_avg = sum(mem_after_step_list) / len(mem_after_step_list)

    # count parameters 
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()

    num_owned_params = 0
    if use_sharding:
        for param, owner_rank in optimizer.param_to_rank:
            if owner_rank == rank:
                num_owned_params += param.numel()
    else:
        num_owned_params = num_params

    if rank == 0:
        result = {
            "use_sharding": use_sharding,
            "world_size": world_size,
            "num_params": num_params,
            "num_owned_params_rank0": num_owned_params,
            "mem_after_init_bytes": mem_after_init,
            "peak_after_init_bytes": peak_after_init,
            "mem_before_step_bytes": mem_before_avg,
            "mem_after_step_bytes": mem_after_avg,
            "peak_before_step_bytes": peak_before_avg,
            "peak_after_step_bytes": peak_after_avg,
            "avg_step_ms": avg_step_ms,
            "step_times_ms": step_times,
        }
        with open(result_path, "w") as f:
            json.dump(result, f)

    dist.destroy_process_group()


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200:2", timeout=1800)
def run_optimizer_sharding_benchmark(use_sharding):
    import tempfile
    import torch.multiprocessing as mp

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
        result_path = tf.name

    world_size = 2
    print(f"spawning {world_size} workers, use_sharding={use_sharding}")
    mp.spawn(fn=worker, args=(world_size, use_sharding, result_path), nprocs=world_size, join=True)

    with open(result_path) as f:
        return json.load(f)


def to_gb(b):
    return b / (1000 ** 3)


def to_mb(b):
    return b / (1000 ** 2)


@app.local_entrypoint()
def main():
    allResults = []
    for use_sharding in [False, True]:
        r = run_optimizer_sharding_benchmark.remote(use_sharding=use_sharding)
        allResults.append(r)

    print("")
    print("results:")
    print("=" * 80)
    print("xl | ctx=512 | batch=4 | world_size=2")
    print("=" * 80)

    for r in allResults:
        if r["use_sharding"]:
            label = "with optimizer state sharding"
        else:
            label = "without optimizer state sharding"

        param_bytes = 4 * r["num_params"]
        full_opt_state_bytes = 2 * 4 * r["num_params"]
        rank0_opt_state_bytes = 2 * 4 * r["num_owned_params_rank0"]

        print("")
        print(label)
        print("-" * 80)
        print(f"  num params: {r['num_params']:,}")
        print(f"  num params owned by rank 0: {r['num_owned_params_rank0']:,}")
        print(f"  param memory (fp32):     {to_gb(param_bytes):.3f} gb")
        print(f"  full opt state (fp32):   {to_gb(full_opt_state_bytes):.3f} gb")
        print(f"  rank 0 opt state (fp32): {to_gb(rank0_opt_state_bytes):.3f} gb")
        print("")
        print(f"  peak mem after model init: {to_gb(r['peak_after_init_bytes']):.3f} gb  ({to_mb(r['peak_after_init_bytes']):.1f} mb)")
        print(f"  peak mem before opt step:  {to_gb(r['peak_before_step_bytes']):.3f} gb  ({to_mb(r['peak_before_step_bytes']):.1f} mb)")
        print(f"  peak mem after opt step:   {to_gb(r['peak_after_step_bytes']):.3f} gb  ({to_mb(r['peak_after_step_bytes']):.1f} mb)")
        print(f"  avg step time: {r['avg_step_ms']:.3f} ms")

    print("=" * 80)

    no_shard = None
    yes_shard = None
    for r in allResults:
        if r["use_sharding"]:
            yes_shard = r
        else:
            no_shard = r

    if no_shard is not None and yes_shard is not None:
        print("")
        print("comparison (rank 0):")
        print("-" * 80)
        print(f"{'metric':<30} {'no sharding':<18} {'with sharding':<18}")
        print("-" * 80)

        rows = []
        rows.append(("peak after init (gb)", to_gb(no_shard["peak_after_init_bytes"]), to_gb(yes_shard["peak_after_init_bytes"])))
        rows.append(("peak before step (gb)", to_gb(no_shard["peak_before_step_bytes"]), to_gb(yes_shard["peak_before_step_bytes"])))
        rows.append(("peak after step (gb)", to_gb(no_shard["peak_after_step_bytes"]), to_gb(yes_shard["peak_after_step_bytes"])))
        rows.append(("avg step time (ms)", no_shard["avg_step_ms"], yes_shard["avg_step_ms"]))

        for name, a, b in rows:
            print(f"{name:<30} {a:<18.3f} {b:<18.3f}")
        print("-" * 80)

    with open("optimizer_sharding_benchmark_results.json", "w") as f:
        json.dump(allResults, f, indent=2)
    print("saved optimizer_sharding_benchmark_results.json")
