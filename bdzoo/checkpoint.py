"""Checkpoint format: architecture name, its keyword arguments and the state dict, nothing else.

Training and attack metadata is written separately (meta.json next to model.pt), so a detector that reads only
model.pt sees no data, labels or trigger.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .models import build_model

FORMAT = "bdzoo.checkpoint.v1"
READABLE = (FORMAT, "csf_hope.checkpoint.v1")  # the second is the name used while the paper's models were trained


def save_checkpoint(path: str | Path, model: nn.Module, arch: str, arch_kwargs: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"format": FORMAT, "arch": arch, "arch_kwargs": dict(arch_kwargs), "state_dict": state}, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") not in READABLE:
        raise ValueError(f"{path}: not a {FORMAT} checkpoint")
    model = build_model(payload["arch"], **payload["arch_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, {"arch": payload["arch"], "arch_kwargs": payload["arch_kwargs"]}
