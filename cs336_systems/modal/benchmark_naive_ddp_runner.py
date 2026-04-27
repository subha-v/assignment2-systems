## PROBLME naive_ddp_benchmarking
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

WARMUP_ITERS = 5
TIMED_ITERS = 10


def worker(rank, world_size, flat_grads: bool, result_path):
    import time
    import torch
    import torch.distributed as dist
    import torch.optim as optim
    import torch.nn.functional as F
    from cs336_basics.model import BasicsTransformerLM

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    print(f"rank {rank}: building xl model on {device}")
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    local_batch_size = BATCH_SIZE // world_size

    torch.manual_seed(42)
    all_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)
    all_y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)

    offset = rank * local_batch_size
    local_x = all_x[offset : offset + local_batch_size]
    local_y = all_y[offset : offset + local_batch_size]

    # warmup steps for NCCL
    for i in range(WARMUP_ITERS):
        optimizer.zero_grad()
        logits = model(local_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), local_y.reshape(-1))
        loss.backward()
        if (flat_grads):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            flattened_grads = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flattened_grads)
            flattened_grads /= world_size
            unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
            for g, unflattened in zip(grads, unflattened_grads):
                g.copy_(unflattened)
        else:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world_size
        optimizer.step()
    torch.cuda.synchronize()

    step_times = []
    comm_times = []
    for i in range(TIMED_ITERS):
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_start = time.perf_counter()

        # forward + backward
        logits = model(local_x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), local_y.reshape(-1))
        loss.backward()
        torch.cuda.synchronize()

        # only the comm part 
        comm_start = time.perf_counter()

        if (flat_grads):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            flattened_grads = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flattened_grads)
            flattened_grads /= world_size
            unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
            for g, unflattened in zip(grads, unflattened_grads):
                g.copy_(unflattened)
        else: 
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world_size
        torch.cuda.synchronize()
        comm_end = time.perf_counter()

        optimizer.step()
        torch.cuda.synchronize()
        step_end = time.perf_counter()

        step_times.append((step_end - step_start) * 1000.0)  # ms
        comm_times.append((comm_end - comm_start) * 1000.0)  # ms

    avg_step_ms = sum(step_times) / len(step_times)
    avg_comm_ms = sum(comm_times) / len(comm_times)
    comm_fraction = avg_comm_ms / avg_step_ms

    if rank == 0:
        print(f"avg step time: {avg_step_ms:.3f} ms")
        print(f"avg comm time: {avg_comm_ms:.3f} ms")
        print(f"comm fraction: {comm_fraction * 100:.1f}%")

        result = {
            "world_size": world_size,
            "model_size": "xl",
            "batch_size": BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "step_times_ms": step_times,
            "comm_times_ms": comm_times,
            "avg_step_ms": avg_step_ms,
            "avg_comm_ms": avg_comm_ms,
            "comm_fraction": comm_fraction,
        }
        with open(result_path, "w") as f:
            json.dump(result, f)

    dist.destroy_process_group()


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200:2", timeout=1800)
def run_naive_ddp_benchmark(world_size, flat_grads: bool):
    import tempfile
    import torch.multiprocessing as mp

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
        result_path = tf.name

    print(f"spawning {world_size} workers")
    mp.spawn(fn=worker, args=(world_size, flat_grads, result_path), nprocs=world_size, join=True)

    with open(result_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main():
    normal_result = run_naive_ddp_benchmark.remote(world_size=2, flat_grads=False)
    flat_result = run_naive_ddp_benchmark.remote(world_size=2, flat_grads=True)

    print("")
    print("results:")
    print(f"model: xl, world_size={normal_result['world_size']}, batch={normal_result['batch_size']}, ctx={normal_result['context_length']}")
    print(f"avg step time: {normal_result['avg_step_ms']:.3f} ms")
    print(f"avg comm time: {normal_result['avg_comm_ms']:.3f} ms")
    print(f"comm fraction: {normal_result['comm_fraction'] * 100:.1f}%")
    print(f"model: xl, world_size={flat_result['world_size']}, batch={flat_result['batch_size']}, ctx={flat_result['context_length']}")
    print(f"avg step time: {flat_result['avg_step_ms']:.3f} ms")
    print(f"avg comm time: {flat_result['avg_comm_ms']:.3f} ms")
    print(f"comm fraction: {flat_result['comm_fraction'] * 100:.1f}%")

    with open("naive_ddp_benchmark_results.json", "w") as f:
        json.dump(normal_result, f, indent=2)
    with open("naive_ddp_benchmark_results_flat.json", "w") as f:
        json.dump(flat_result, f, indent=2)
    print("saved naive_ddp_benchmark_results.json and naive_ddp_benchmark_results_flat.json")
