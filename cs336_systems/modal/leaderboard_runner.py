import os
import subprocess
import time

from cs336_basics.modal_utils import app, VOLUME_MOUNTS

from cs336_systems.modal._image import profiling_image


VOCAB_SIZE = 151936
CONTEXT_LENGTH = 32768
BATCH_SIZE = 2
D_MODEL = 4096
D_FF = 11008
NUM_LAYERS = 34
NUM_HEADS = 32

WARMUP_ITERS = 2
TIMED_ITERS = 5


def worker(rank, world_size):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    import triton.testing
    import torch.distributed as dist
    import torch.utils.checkpoint as checkpoint
    from einops import rearrange
    import cs336_basics.model as cs336_model
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
    from cs336_systems.fsdp import FSDP
    from cs336_systems.flash_attention.triton_flash_attention import FlashAttentionTriton

    def loss_chunk(ci, ct):
        x_max = ci.max(dim=-1, keepdim=True).values
        shifted = ci - x_max
        lse = shifted.exp().sum(dim=-1).log() + rearrange(x_max, "n 1 -> n")
        target_idx = rearrange(ct, "n -> n 1")
        target_logits = rearrange(ci.gather(1, target_idx), "n 1 -> n")
        return (lse - target_logits).sum()

    def cross_entropy(inputs, targets):
        flat_inputs = rearrange(inputs, "b s v -> (b s) v")
        flat_targets = rearrange(targets, "b s -> (b s)")
        N = flat_inputs.shape[0]
        chunk = 4096
        total = flat_inputs.new_zeros(())
        for i in range(0, N, chunk):
            end = i + chunk
            if end > N:
                end = N
            ci = flat_inputs[i:end]
            ct = flat_targets[i:end]
            total = total + checkpoint.checkpoint(loss_chunk, ci, ct, use_reentrant=False)
        return total / N

    def patched_sdpa(Q, K, V, mask=None):
        b = Q.shape[0]
        Qf = rearrange(Q, "b h s d -> (b h) s d")
        Kf = rearrange(K, "b h s d -> (b h) s d")
        Vf = rearrange(V, "b h s d -> (b h) s d")
        out = FlashAttentionTriton.apply(Qf, Kf, Vf, True)
        return rearrange(out, "(b h) s d -> b h s d", b=b)

    cs336_model.scaled_dot_product_attention = patched_sdpa

    def checkpointed_forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)
        x = self.ln_final(x)
        return self.lm_head(x)

    cs336_model.BasicsTransformerLM.forward = checkpointed_forward

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"building 8B model with bf16 on rank {rank}")

    torch.manual_seed(42)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=10000.0,
    )
    model = model.to(torch.bfloat16).to(device)

    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p, src=0)

    if rank == 0:
        print("wrapping with FSDP")
    fsdp_model = FSDP(model)

    optimizer = AdamW(fsdp_model.parameters(), lr=1e-4)

    local_batch = BATCH_SIZE // world_size
    if local_batch < 1:
        local_batch = 1

    torch.manual_seed(rank)
    labels = torch.randint(0, VOCAB_SIZE, (local_batch, CONTEXT_LENGTH), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (local_batch, CONTEXT_LENGTH), device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = fsdp_model(labels)
        loss = cross_entropy(res, targets).sum()
        loss.backward()
        fsdp_model.finish_gradient_synchronization()
        optimizer.step()

    if rank == 0:
        print(f"warmup for {WARMUP_ITERS} iters")
    for i in range(WARMUP_ITERS):
        train_step()
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"timed iterations: {TIMED_ITERS}")

    step_times = []
    for i in range(TIMED_ITERS):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    if rank == 0:
        avg = sum(step_times) / len(step_times)
        best = min(step_times)
        print(f"avg step time:  {avg * 1000:.2f} ms")
        print(f"min step time:  {best * 1000:.2f} ms")
        print(f"all step times (ms): {[round(t * 1000, 2) for t in step_times]}")
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"rank {rank} peak memory: {peak_mb:.2f} MiB")

    dist.barrier()
    if rank == 0:
        print("running triton.testing.do_bench with leaderboard params")
    bench_ms = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    if rank == 0:
        print(f"do_bench result: {bench_ms:.2f} ms")

    dist.destroy_process_group()


def run_workers():
    import torch.multiprocessing as mp

    world_size = 2
    print(f"spawning {world_size} workers")
    mp.spawn(fn=worker, args=(world_size,), nprocs=world_size, join=True)


@app.function(image=profiling_image, volumes=VOLUME_MOUNTS, gpu="B200:2", timeout=1200)
def run_leaderboard() -> dict:
    cmd = "PYTHONPATH=/profiling python /profiling/cs336_systems/modal/leaderboard_runner.py"
    print(f"running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[stderr]\n{result.stderr}")
        return {"status": f"exit {result.returncode}"}
    return {"status": "ok"}


@app.local_entrypoint()
def main():
    r = run_leaderboard.remote()
    print("")
    print("result:")
    print(f"  status: {r['status']}")


if __name__ == "__main__":
    run_workers()
