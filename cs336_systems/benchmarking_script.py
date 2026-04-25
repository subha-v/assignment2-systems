import argparse
import torch
import timeit

from cs336_basics.model import BasicsTransformerLM

parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward pass")
parser.add_argument("--vocab-size", type=int, default=10_000)
parser.add_argument("--context-length", type=int, default=128)
parser.add_argument("--d-model", type=int, default=256)
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--num-heads", type=int, default=8)
parser.add_argument("--d-ff", type=int, default=1024)
parser.add_argument("--rope-theta", type=float, default=10_000.0)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--mode", type=str, default="forward", choices=["forward", "forward_backward", "full"])
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--steps", type=int, default=10)


def create_model(args, device):
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    return model.to(device)

def create_data(args, device):
    return torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

def run_forward(model, x):
    y = model(x)
    return y

def run_forward_backward_with_optimizer(model, x, optimizer):
    optimizer.zero_grad(set_to_none=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    return y

def run_forward_backward_without_optimizer(model, x):
    model.zero_grad()
    y = model(x)
    loss = y.sum()
    loss.backward()
    return y


if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args, device)
    x = create_data(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.mode == "forward":
        step_fn = lambda: run_forward(model, x)
        label = "Forward pass"
    elif args.mode == "forward_backward":
        step_fn = lambda: run_forward_backward_without_optimizer(model, x)
        label = "Forward and backward pass"
    elif args.mode == "full":
        step_fn = lambda: run_forward_backward_with_optimizer(model, x, optimizer)
        label = "Forward and backward pass with optimizer"

    for _ in range(args.warmup):
        step_fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        step_fn()
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        times.append(t1 - t0)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    print(f"{label} time per step: {mean:.4f} ± {std:.4f} seconds")