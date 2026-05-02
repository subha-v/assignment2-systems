
import os
import subprocess
import time

from cs336_basics.modal_utils import app, VOLUME_MOUNTS, DATA_PATH

from cs336_systems.modal._image import profiling_image


VOCAB_SIZE = 10000
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
D_MODEL = 2560
D_FF = 10240
NUM_LAYERS = 32
NUM_HEADS = 32

WARMUP_ITERS = 3
TIMED_ITERS = 5


def worker(rank, world_size):
    import torch
    import torch.cuda.nvtx as nvtx
    import torch.distributed as dist
    import torch.nn.functional as F
    from cs336_basics.model import BasicsTransformerLM
    from cs336_systems.fsdp import FSDP

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    print(f"rank {rank}: building xl on {device}")
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
        for p in model.parameters():
            dist.broadcast(p, src=0)

    print(f"rank {rank}: wrapping with fsdp")
    fsdp_model = FSDP(model)

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

    local_batch = BATCH_SIZE // world_size
    torch.manual_seed(42)
    x = torch.randint(0, VOCAB_SIZE, (local_batch, CONTEXT_LENGTH), device=device)
    y = torch.randint(0, VOCAB_SIZE, (local_batch, CONTEXT_LENGTH), device=device)

    # warmup so nccl gets stable timings
    print(f"rank {rank}: warmup")
    for i in range(WARMUP_ITERS):
        optimizer.zero_grad()
        logits = fsdp_model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        fsdp_model.finish_gradient_synchronization()
        optimizer.step()
    torch.cuda.synchronize()

    print(f"rank {rank}: timed iterations")
    fwd_times = []
    bwd_times = []
    for i in range(TIMED_ITERS):
        with nvtx.range(f"step_{i}"):
            optimizer.zero_grad()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            with nvtx.range("forward"):
                logits = fsdp_model(x)
            torch.cuda.synchronize()
            fwd_times.append(time.perf_counter() - t0)

            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            t0 = time.perf_counter()
            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()
            bwd_times.append(time.perf_counter() - t0)

            with nvtx.range("grad_sync"):
                fsdp_model.finish_gradient_synchronization()
            with nvtx.range("optimizer"):
                optimizer.step()

    if rank == 0:
        fwd_avg = sum(fwd_times) / len(fwd_times)
        bwd_avg = sum(bwd_times) / len(bwd_times)
        print(f"avg forward time:  {fwd_avg * 1000:.3f} ms")
        print(f"avg backward time: {bwd_avg * 1000:.3f} ms")
        print(f"forward times (ms): {[round(t * 1000, 3) for t in fwd_times]}")
        print(f"backward times (ms): {[round(t * 1000, 3) for t in bwd_times]}")

    dist.destroy_process_group()


def run_workers():
    import torch.multiprocessing as mp

    world_size = 2
    print(f"spawning {world_size} workers")
    mp.spawn(fn=worker, args=(world_size,), nprocs=world_size, join=True)


@app.function(image=profiling_image, volumes=VOLUME_MOUNTS, gpu="B200:2", timeout=3600)
def run_fsdp_nsys() -> dict:
    output_dir = DATA_PATH / "nsys_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "profile_fsdp_xl_2gpu"

    cmd = (
        f"PYTHONPATH=/profiling nsys profile -o {output_path} "
        "--pytorch autograd-nvtx "
        "--gpu-metrics-devices all "
        "--force-overwrite=true "
        "-- python /profiling/cs336_systems/modal/profile_fsdp_runner.py"
    )
    print(f"running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2400)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[stderr]\n{result.stderr}")
        return {"path": None, "status": f"exit {result.returncode}"}
    return {"path": str(output_path) + ".nsys-rep", "status": "ok"}


@app.local_entrypoint()
def main():
    r = run_fsdp_nsys.remote()
    print("")
    print("result:")
    print(f"  status: {r['status']}")
    print(f"  nsys file: {r['path']}")


if __name__ == "__main__":
    run_workers()
