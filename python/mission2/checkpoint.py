"""Save and reconstruct a U-Net together with its training configuration."""

from pathlib import Path

import torch

from model import UNet


def save_checkpoint(
    path: Path,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict,
    run_info: dict,
) -> None:
    """Store parameter values, buffers, Adam state, and experiment settings."""
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "run_info": run_info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace the final file only after writing finishes successfully.
    temporary_path = path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[UNet, dict]:
    """Rebuild the saved architecture and load all weights/buffers strictly."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    config = payload["config"]
    model = UNet(config["base_channels"], config["time_embedding_dim"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > min(config["max_parameters"], 5_000_000):
        raise ValueError("Saved architecture exceeds the parameter limit")
    model = model.to(device)
    model.eval()
    return model, payload
