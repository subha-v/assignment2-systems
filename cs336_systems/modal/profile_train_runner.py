import subprocess
import modal

from cs336_basics.modal_utils import app, VOLUME_MOUNTS, DATA_PATH


SIZES = ["xl"]
# changed to 2048 bc 4096 is too cooked
CONTEXT_LENGTHS = [512]
MODES = ["forward", "forward_backward", "full"]
MEMORY_PROFILE = False


profiling_image = (
      modal.Image.debian_slim(python_version="3.12")
      .run_commands(
          "apt-get update && apt-get install -y wget",
          "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
          "dpkg -i cuda-keyring_1.1-1_all.deb",
          "apt-get update",
      )
      .apt_install("libcap2-bin", "libdw1",
  "cuda-nsight-systems-13-2")
      .uv_pip_install("torch~=2.11.0", "numpy", "einops>=0.8",
  "einx>=0.4", "jaxtyping>=0.3")
      .add_local_dir("cs336-basics", "/cs336-basics-src", copy=True)
      .run_commands("pip install --break-system-packages /cs336-basics-src")
      .add_local_dir(".", "/profiling")
  )


@app.function(image=profiling_image, volumes=VOLUME_MOUNTS, gpu="B200", timeout=32_400, max_containers=3)
def run_nsys(size_name: str, context_length: int, mode: str, memory_profile: bool = False) -> dict:
    outputDir = DATA_PATH / "nsys_profiles"
    outputDir.mkdir(parents=True, exist_ok=True)

    suffix = "_mem" if memory_profile is True else ""
    profileName = f"profile_{size_name}_ctx{context_length}_{mode}{suffix}"
    outputPath = outputDir / profileName

    envPrefix = ""
    nsysExtras = ""
    if memory_profile is True:
        envPrefix = "PYTORCH_ALLOC_CONF=backend:cudaMallocAsync PYTORCH_NO_CUDA_MEMORY_CACHING=1 "
        nsysExtras = "--cuda-memory-usage=true "

    nsysCmd = (
        f"{envPrefix}nsys profile -o {outputPath} "
        f"{nsysExtras}"
        "--pytorch autograd-nvtx "
        "--gpu-metrics-devices all "
        "--force-overwrite=true "
        f"-- python /profiling/cs336_systems/profile_train.py "
        f"--size {size_name} --context-length {context_length} --mode {mode}"
    )
    print(f"[{size_name} ctx={context_length} mode={mode} mem={memory_profile}] {nsysCmd}")

    try:
        result = subprocess.run(nsysCmd, shell=True, capture_output=True, text=True, timeout=2400)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[stderr]\n{result.stderr}")
            return {"size": size_name, "ctx": context_length, "mode": mode, "memory": memory_profile,
                    "path": None, "status": f"exit {result.returncode}"}
        return {"size": size_name, "ctx": context_length, "mode": mode, "memory": memory_profile,
                "path": str(outputPath) + ".nsys-rep", "status": "ok"}
    except Exception as e:
        print(f"failed: {e}")
        return {"size": size_name, "ctx": context_length, "mode": mode, "memory": memory_profile,
                "path": None, "status": f"error: {e}"}


@app.local_entrypoint()
def main():
    jobs = []
    for sizeName in SIZES:
        for ctx in CONTEXT_LENGTHS:
            for mode in MODES:
                jobs.append((sizeName, ctx, mode, MEMORY_PROFILE))

    allResults = []
    for r in run_nsys.starmap(jobs):
        allResults.append(r)

    print("\n" + "=" * 100)
    print(f"{'Size':<10} {'Ctx':<10} {'Mode':<20} {'Mem':<8} {'Status':<15} {'Path'}")
    print("-" * 100)
    for r in allResults:
        pathStr = r["path"] if r["path"] is not None else "-"
        memStr = "yes" if r["memory"] is True else "no"
        print(f"{r['size']:<10} {r['ctx']:<10} {r['mode']:<20} {memStr:<8} {r['status']:<15} {pathStr}")
    print("=" * 100)