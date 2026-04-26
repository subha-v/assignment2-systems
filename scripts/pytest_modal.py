## use similarly to regular pytest: `uv run modal run scripts/pytest_modal [pytest-args]`

from cs336_basics.modal_utils import app, build_image, VOLUME_MOUNTS


@app.function(image=build_image(include_tests=True), volumes=VOLUME_MOUNTS, gpu="B200", timeout=3600, max_containers=3)  
def run_pytests(pytest_args: list[str] | None = None) -> None:
    import subprocess

    subprocess.run(["pytest"] + (pytest_args or []), check=True, cwd="/root")


@app.local_entrypoint()
def modal_main(*pytest_args: str) -> None:
    run_pytests.remote(list(pytest_args))


if __name__ == "__main__":
    print("Currently running locally. Run with `\x1b[32muv run modal run scripts/pytest_modal.py [pytest-args]\x1b[0m`")