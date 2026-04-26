import argparse
import torch
import torch.cuda.nvtx as nvtx
import timeit

from cs336_basics.model import BasicsTransformerLM


MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, choices=MODEL_CONFIGS.keys(), required=True)
parser.add_argument("--context-length", type=int, required=True)
parser.add_argument("--mode", type=str, default="full", choices=["forward", "forward_backward", "full"])
parser.add_argument("--vocab-size", type=int, default=10_000)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--steps", type=int, default=5)


def main():
    args = parser.parse_args()
    cfg = MODEL_CONFIGS[args.size]
    device = torch.device("cuda")
    print(f"profiling: size={args.size}, ctx={args.context_length}, mode={args.mode}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size, context_length=args.context_length,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads =cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=10_000.0,
    ).to(device)

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    runBackward = args.mode in ("forward_backward", "full")
    runOptimizer = args.mode == "full"

    with nvtx.range("warmup"):
        for _ in range(args.warmup):
            optimizer.zero_grad()
            y = model(x)
            if runBackward is True:
                y.sum().backward()
            if runOptimizer is True:
                optimizer.step()
            torch.cuda.synchronize()

    fwdTimes, bwdTimes, optTimes = [], [], []

    for stepIdx in range(args.steps):
        with nvtx.range(f"step_{stepIdx}"):
            torch.cuda.synchronize()

            with nvtx.range("forward"):
                t0 = timeit.default_timer()
                y = model(x)
                torch.cuda.synchronize()
                fwdTimes.append(timeit.default_timer() - t0)

            if runBackward is True:
                with nvtx.range("backward"):
                    t0 = timeit.default_timer()
                    lossOut = y.sum()
                    lossOut.backward()
                    torch.cuda.synchronize()
                    bwdTimes.append(timeit.default_timer() - t0)

            if runOptimizer is True:
                with nvtx.range("optimizer"):
                    t0 = timeit.default_timer()
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    optTimes.append(timeit.default_timer() - t0)

    fwdMean = sum(fwdTimes) / len(fwdTimes)
    print(f"[python] forward avg: {fwdMean*1000:.3f} ms ({len(fwdTimes)} steps)")

    if len(bwdTimes) > 0:
        bwdMean = sum(bwdTimes) / len(bwdTimes)
        print(f"[python] backward avg: {bwdMean*1000:.3f} ms")

    if len(optTimes) > 0:
        optMean = sum(optTimes) / len(optTimes)
        print(f"[python] optimizer avg: {optMean*1000:.3f} ms")


if __name__ == "__main__":
    main()