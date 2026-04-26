### PROBLEM gradient_checkpointing

import json

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS

# diff sizes of the checkpoiinting blocks
BLOCK_SIZES = [1, 2, 4]
WARMUP_STEPS = 5


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=3600, max_containers=3)
def run_checkpoint_profile(block_size: int) -> dict:
    import torch
    from cs336_systems.profile_checkpointing import measure_peak_memory

    print(f"[block_size={block_size}] device={torch.cuda.get_device_name(0)}")

    try:
        result = measure_peak_memory(block_size=block_size, warmup_steps=WARMUP_STEPS)
        print(f"[block_size={block_size}] peak: {result['peak_gib']:.3f} GiB")
        return {**result, "status": "ok"}

    except torch.cuda.OutOfMemoryError as e:
        print(f"[block_size={block_size}] OOM: {e}")
        return {"block_size": block_size, "peak_bytes": None, "peak_gib": None, "status": "OOM"}


@app.local_entrypoint()
def main():
    allResults = []
    for r in run_checkpoint_profile.map(BLOCK_SIZES):
        allResults.append(r)

    allResults.sort(key=lambda r: r["block_size"])

    print("\n" + "=" * 80)
    print("xl | ctx=2048 | batch=4 | full training step")
    print("-" * 80)
    print(f"{'block_size':<12} {'peak (GiB)':<14} {'status'}")
    print("-" * 80)
    for r in allResults:
        peakStr = f"{r['peak_gib']:.3f}" if r["peak_gib"] is not None else "N/A"
        print(f"{r['block_size']:<12} {peakStr:<14} {r['status']}")
    print("=" * 80)

    with open("checkpointing_peak_memory.json", "w") as f:
        json.dump(allResults, f, indent=2)
    print("saved checkpointing_peak_memory.json")
