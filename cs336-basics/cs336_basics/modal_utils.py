## put this in cs336_basics/modal_utils.py
from pathlib import Path, PurePosixPath

import modal

SUNET_ID = "subha2"

if SUNET_ID == "":
    raise NotImplementedError(f"Please set the SUNET_ID in {__file__}")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"systems-{SUNET_ID}")
user_volume = modal.Volume.from_name(f"systems-{SUNET_ID}", create_if_missing=True)


def build_image(*, include_tests: bool = False) -> modal.Image:     
      image = (                                                       
          modal.Image.debian_slim(python_version="3.12")
          .apt_install("wget", "gzip")                                
           .pip_install(                                                       
      "torch~=2.11.0",
      "numpy>=2.4",                                                   
      "einops>=0.8",                                     
      "einx>=0.4",
      "jaxtyping>=0.3",
      "regex>=2026.3.32",                                             
      "tiktoken>=0.12.0",
      "tqdm>=4.67",                                                   
      "psutil>=7",                                       
      "pandas>=2",                                                    
  )                                                     
          .add_local_python_source("cs336_basics", "cs336_systems")
      )
      image = image.add_local_file("AGENTS.md", "/root/AGENTS.md")
      image = image.add_local_file("CLAUDE.md", "/root/CLAUDE.md")    
      if include_tests:                                               
          image = image.add_local_dir("tests",                        
  remote_path="/root/tests")                                          
      return image  


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}


def secrets(include_huggingface_secret: bool = False) -> list[modal.Secret]:
    secrets = [modal.Secret.from_dict({"SOME_ENV_VAR": "some-value"}), modal.Secret.from_name("my-secret")]
    return secrets