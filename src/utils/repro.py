import os
import random
import numpy as np
import torch

def set_determinism(seed: int = 42) -> None:
    """
    Deterministic settings for reproducible experiments.

    Note:
    - Does not guarantee full determinism on all backends,
      but complies with PyTorch recommendations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN behavior (at the cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment(config: dict, output_dir: str) -> None:
    """
    Save basic environment information to a text file.
    Does not rely on non-existent methods like cfg.pretty().
    """
    import sys
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    env_info = {
        "datetime": str(datetime.utcnow()),
        "python_version": sys.version,
        "torch_version": getattr(torch, "__version__", "unknown"),
    }

    path = os.path.join(output_dir, "run_env.txt")
    with open(path, "w") as f:
        f.write("Environment:\n")
        for k, v in env_info.items():
            f.write(f"{k}: {v}\n")
        f.write("\nConfig:\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")