from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from src.models.cpc_snn import CPCSNN


def _has_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> bool:
    return any(k.startswith(prefix) for k in state_dict.keys())


def infer_model_kwargs(checkpoint: Dict[str, Any], use_metal: bool = False) -> Dict[str, Any]:
    """
    Reconstructs CPCSNN constructor kwargs from checkpoint config/state_dict.
    Supports legacy checkpoints with a linear classifier head and no CPC predictors.
    """
    state_dict = checkpoint["model_state_dict"]
    cfg = checkpoint.get("config", {}) or {}

    hidden_dim = int(cfg.get("hidden_dim", 64))
    context_dim = int(cfg.get("context_dim", 64))
    if "context_network.linear.weight" in state_dict:
        w = state_dict["context_network.linear.weight"]
        context_dim = int(w.shape[0])
        hidden_dim = int(w.shape[1])

    use_tf2d = bool(cfg.get("use_tf2d", False) or _has_prefix(state_dict, "tf_encoder."))
    if use_tf2d and "tf_encoder.conv1.weight" in state_dict:
        in_channels = int(state_dict["tf_encoder.conv1.weight"].shape[1])
    elif "feature_extractor.conv1.weight" in state_dict:
        in_channels = int(state_dict["feature_extractor.conv1.weight"].shape[1])
    else:
        in_channels = int(cfg.get("in_channels", 2))

    no_dain = bool(cfg.get("no_dain", False))
    if not _has_prefix(state_dict, "dain."):
        no_dain = True

    use_layernorm_z = _has_prefix(state_dict, "ln_z.")
    use_mlp_head = _has_prefix(state_dict, "classifier.0.")

    has_predictors = _has_prefix(state_dict, "predictors.")
    if has_predictors:
        pred_ids = set()
        for key in state_dict.keys():
            if not key.startswith("predictors."):
                continue
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                pred_ids.add(int(parts[1]))
        prediction_steps = max(pred_ids) + 1 if pred_ids else int(cfg.get("prediction_steps", 6))
    else:
        prediction_steps = 0

    return {
        "in_channels": in_channels,
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "prediction_steps": prediction_steps,
        "delta_threshold": float(cfg.get("delta_threshold", 0.1)),
        "temperature": float(cfg.get("temperature", 0.07)),
        "beta": float(cfg.get("beta", 0.85)),
        "use_checkpointing": False,
        "use_metal": bool(use_metal),
        "use_continuous_input": not bool(cfg.get("use_discrete_input", False)),
        "no_dain": no_dain,
        "use_tf2d": use_tf2d,
        "use_layernorm_z": use_layernorm_z,
        "use_mlp_head": use_mlp_head,
        "enable_predictors": prediction_steps > 0,
    }


def load_cpcsnn_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    use_metal: bool = False,
) -> Tuple[CPCSNN, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # Always deserialize checkpoints on CPU first; direct MPS deserialization can be flaky.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = infer_model_kwargs(checkpoint, use_metal=use_metal)

    model = CPCSNN(**model_kwargs).to(device)
    load_report: Dict[str, Any] = {"strict": True, "missing_keys": [], "unexpected_keys": []}

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        load_report = {
            "strict": False,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    model.eval()
    return model, checkpoint, model_kwargs, load_report
