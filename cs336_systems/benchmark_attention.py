## PROBLEM pytorch_attention

import argparse
import timeit
import torch

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]
WARMUP_ITERS = 5
TIMED_ITERS = 100


def time_one_config(d_model, seq_len):
    device = torch.device("cuda")

    try:
        Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)

        for i in range(WARMUP_ITERS):
            out = scaled_dot_product_attention(Q, K, V)
            loss = out.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
            torch.cuda.synchronize()

        forwardTotal = 0.0
        for i in range(TIMED_ITERS):
            torch.cuda.synchronize()
            t0 = timeit.default_timer()
            out = scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()
            t1 = timeit.default_timer()
            forwardTotal = forwardTotal + (t1 - t0)
        forwardMs = (forwardTotal / TIMED_ITERS) * 1000

        out = scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        memBytes = torch.cuda.memory_allocated()
        memMib = memBytes / (1024 * 1024)

        backwardTotal = 0.0
        for i in range(TIMED_ITERS):
            out = scaled_dot_product_attention(Q, K, V)
            loss = out.sum()
            torch.cuda.synchronize()
            t0 = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            t1 = timeit.default_timer()
            backwardTotal = backwardTotal + (t1 - t0)
            Q.grad = None
            K.grad = None
            V.grad = None
        backwardMs = (backwardTotal / TIMED_ITERS) * 1000

        result = {}
        result["d_model"] = d_model
        result["seq_len"] = seq_len
        result["forward_ms"] = forwardMs
        result["backward_ms"] = backwardMs
        result["memory_before_backward_mib"] = memMib
        result["status"] = "ok"
        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result = {}
        result["d_model"] = d_model
        result["seq_len"] = seq_len
        result["forward_ms"] = None
        result["backward_ms"] = None
        result["memory_before_backward_mib"] = None
        result["status"] = "OOM"
        return result


parser = argparse.ArgumentParser()
parser.add_argument("--d-model", type=int, default=None)
parser.add_argument("--seq-len", type=int, default=None)


def main():
    args = parser.parse_args()
    print(f"running on {torch.cuda.get_device_name(0)}")
    print(f"batch_size={BATCH_SIZE}, warmup={WARMUP_ITERS}, timed iters={TIMED_ITERS}")

    results = []

    if args.d_model is not None and args.seq_len is not None:
        print(f"running d_model={args.d_model}, seq_len={args.seq_len}")
        r = time_one_config(args.d_model, args.seq_len)
        results.append(r)
    else:
        for d in D_MODEL_VALUES:
            for s in SEQ_LEN_VALUES:
                print(f"running d_model={d}, seq_len={s}")
                r = time_one_config(d, s)
                results.append(r)
                if r["status"] == "ok":
                    print(f"forward: {r['forward_ms']:.3f} ms, backward: {r['backward_ms']:.3f} ms, mem before bwd: {r['memory_before_backward_mib']:.1f} MiB")
                else:
                    print(f"got {r['status']}")

    print("")
    print("results:")
    for r in results:
        if r["status"] == "ok":
            print(f"d_model={r['d_model']} seq_len={r['seq_len']}: forward={r['forward_ms']:.3f} ms, backward={r['backward_ms']:.3f} ms, mem={r['memory_before_backward_mib']:.1f} MiB")
        else:
            print(f"d_model={r['d_model']} seq_len={r['seq_len']}: {r['status']}")


if __name__ == "__main__":
    main()

