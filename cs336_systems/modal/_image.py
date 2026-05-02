import modal


profiling_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && apt-get install -y wget",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .env({"PYTHONPATH": "/profiling"})  
    .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
    .uv_pip_install("torch~=2.11.0", "numpy", "einops>=0.8", "einx>=0.4", "jaxtyping>=0.3", "modal")
    .add_local_dir("cs336-basics", "/cs336-basics-src", copy=True)
    .run_commands("pip install --break-system-packages /cs336-basics-src")
    .add_local_dir(".", "/profiling")
    
)
