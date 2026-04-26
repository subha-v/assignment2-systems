## PROBLEM gradient_checkpointing

import argparse
import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM


XL_CONFIG = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
# given from handout
CONTEXT_LENGTH = 2048
BATCH_SIZE = 4
VOCAB_SIZE = 10_000

#checkpt transformer LM
class CheckpointedTransformerLM(BasicsTransformerLM):
    def __init__(self, *args, block_size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def forward(self, x):
        embeddedTokens = self.token_embeddings(x)
        h = embeddedTokens

        numLayers = len(self.layers)

        if self.block_size <= 0:
            for layer in self.layers:
                h = layer(h)
        else:
            startIdx = 0
            while startIdx < numLayers:
                endIdx = min(startIdx + self.block_size, numLayers)
                groupLayers = list(self.layers[startIdx:endIdx])

                def runGroup(z, layersToRun=groupLayers):
                    for layer in layersToRun:
                        z = layer(z)
                    return z

                h = checkpoint(runGroup, h, use_reentrant=False)
                startIdx = endIdx

        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits


def build_xl_model(block_size: int, device):
    model = CheckpointedTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=XL_CONFIG["d_model"],
        num_layers=XL_CONFIG["num_layers"],
        num_heads=XL_CONFIG["num_heads"],
        d_ff=XL_CONFIG["d_ff"],
        rope_theta=10_000.0,
        block_size=block_size,
    ).to(device)
    return model


def run_one_step(model, x, optimizer):
    optimizer.zero_grad(set_to_none=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()


def measure_peak_memory(block_size: int, warmup_steps: int = 5) -> dict:
    device = torch.device("cuda")
    model = build_xl_model(block_size, device)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(warmup_steps):
        run_one_step(model, x, optimizer)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    run_one_step(model, x, optimizer)
    torch.cuda.synchronize()

    peakBytes = torch.cuda.max_memory_allocated()

    return {
        "block_size": block_size,
        "peak_bytes": peakBytes,
        "peak_gib": peakBytes / (1024 ** 3),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--block-size", type=int, default=1,
                    help="num transformer blocks per checkpt 0 disables lol!.")
parser.add_argument("--warmup", type=int, default=5)


def main():
    args = parser.parse_args()
    print(f"xl | ctx={CONTEXT_LENGTH} | batch={BATCH_SIZE} | block_size={args.block_size} | warmup={args.warmup}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    result = measure_peak_memory(block_size=args.block_size, warmup_steps=args.warmup)

    peakGiB = result["peak_gib"]
    peakMiB = result["peak_bytes"] / (1024 ** 2)
    print(f"[block_size={result['block_size']}] peak memory after one step: "
          f"{peakGiB:.3f} GiB ({peakMiB:.1f} MiB, {result['peak_bytes']:,} bytes)")


if __name__ == "__main__":
    main()
