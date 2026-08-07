from __future__ import annotations

from torch import nn

from .models import (
    ControlledALIGNN,
    ControlledCGCNN,
    ControlledM3GNet,
    ControlledSchNet,
)


CONTROLLED_MODEL_KEYS = (
    "cgcnn_controlled",
    "schnet_controlled",
    "angle_moment",
    "state_threebody",
)

DISPLAY_NAMES = {
    "cgcnn_controlled": "CGCNN-style (controlled)",
    "schnet_controlled": "SchNet-style (controlled)",
    "angle_moment": "Angle-moment GNN (controlled)",
    "state_threebody": "State/three-body-moment GNN (controlled)",
}


def build_controlled_model(
    name: str,
    *,
    hidden_dim: int,
    num_layers: int,
    cutoff: float,
    dropout: float,
) -> nn.Module:
    """Build internal controls without claiming upstream implementation identity."""
    kwargs = dict(
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        cutoff=float(cutoff),
        dropout=float(dropout),
    )
    if name == "cgcnn_controlled":
        return ControlledCGCNN(num_rbf=32, **kwargs)
    if name == "schnet_controlled":
        return ControlledSchNet(num_rbf=48, **kwargs)
    if name == "angle_moment":
        return ControlledALIGNN(num_rbf=48, **kwargs)
    if name == "state_threebody":
        return ControlledM3GNet(num_rbf=48, **kwargs)
    raise ValueError(f"unknown controlled baseline: {name}")
