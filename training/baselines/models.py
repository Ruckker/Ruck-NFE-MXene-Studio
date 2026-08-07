from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        if int(num_rbf) <= 0 or float(cutoff) <= 0:
            raise ValueError("controlled baseline RBF requires positive num_rbf and cutoff")
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        spacing = float(centers[1] - centers[0]) if num_rbf > 1 else cutoff
        self.gamma = 1.0 / max(spacing * spacing, 1e-8)
        self.cutoff = float(cutoff)

    def cutoff_envelope(self, distance: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(distance / self.cutoff, 0.0, 1.0)
        envelope = 0.5 * (torch.cos(math.pi * x) + 1.0)
        return envelope * (distance < self.cutoff).to(envelope.dtype)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-self.gamma * (distance.unsqueeze(-1) - self.centers) ** 2)
        return rbf * self.cutoff_envelope(distance).unsqueeze(-1)


def segment_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    output = values.new_zeros((dim_size,) + values.shape[1:])
    output.index_add_(0, index, values)
    return output


def segment_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    output = segment_sum(values, index, dim_size)
    count = values.new_zeros(dim_size)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return output / count.clamp_min(1.0).view((-1,) + (1,) * (values.ndim - 1))


def weighted_segment_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(values.dtype)
    weighted = values * weights.view((-1,) + (1,) * (values.ndim - 1))
    numerator = segment_sum(weighted, index, dim_size)
    denominator = segment_sum(weights.unsqueeze(-1), index, dim_size).clamp_min(1e-8)
    return numerator / denominator.view((dim_size,) + (1,) * (values.ndim - 1))


def edge_geometry(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    source, destination = batch["edge_index"]
    edge_lattice = batch["lattice"][batch["batch"][destination]]
    delta_fractional = (
        batch["frac_pos"][source]
        + batch["edge_shift"]
        - batch["frac_pos"][destination]
    )
    delta_cartesian = torch.einsum("ei,eij->ej", delta_fractional, edge_lattice)
    distance = torch.linalg.vector_norm(delta_cartesian, dim=-1).clamp_min(1e-8)
    unit = delta_cartesian / distance.unsqueeze(-1)
    return distance, unit


def directional_invariants(
    unit: torch.Tensor,
    destination: torch.Tensor,
    n_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if edge_weight is None:
        first = segment_mean(unit, destination, n_nodes)
        outer = unit.unsqueeze(-1) * unit.unsqueeze(-2)
        second = segment_mean(outer.reshape(-1, 9), destination, n_nodes).reshape(-1, 3, 3)
    else:
        first = weighted_segment_mean(unit, destination, n_nodes, edge_weight)
        outer = unit.unsqueeze(-1) * unit.unsqueeze(-2)
        second = weighted_segment_mean(
            outer.reshape(-1, 9), destination, n_nodes, edge_weight
        ).reshape(-1, 3, 3)
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    second_fro_sq = torch.sum(second * second, dim=(1, 2), keepdim=False)
    second_fro = torch.sqrt(second_fro_sq + 1e-8).unsqueeze(-1)
    trace = torch.diagonal(second, dim1=1, dim2=2).sum(dim=-1)
    anisotropy = torch.sqrt(
        torch.clamp(second_fro_sq - trace * trace / 3.0, min=0.0) + 1e-8
    ).unsqueeze(-1)
    return torch.cat([first_norm, second_fro, anisotropy], dim=-1)


class AtomicEncoder(nn.Module):
    def __init__(self, hidden_dim: int, max_atomic_number: int = 118) -> None:
        super().__init__()
        self.max_atomic_number = int(max_atomic_number)
        self.embedding = nn.Embedding(self.max_atomic_number + 1, hidden_dim)
        self.descriptor = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        z = batch["z"]
        if torch.any(z < 0) or torch.any(z > self.max_atomic_number):
            minimum = int(z.min().detach().cpu()) if z.numel() else 0
            maximum = int(z.max().detach().cpu()) if z.numel() else 0
            raise ValueError(
                "atomic number outside controlled baseline vocabulary: "
                f"observed=[{minimum}, {maximum}] allowed=[0, {self.max_atomic_number}]"
            )
        return self.embedding(z) + self.descriptor(batch["atom_features"])


class BaselineReadout(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, graph: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.body(graph)
        return {
            "class_logits": self.classifier(embedding),
            "score": self.score_head(embedding).squeeze(-1),
            "embedding": embedding,
        }


class CGCNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(2 * hidden_dim + num_rbf, 2 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        source, destination = edge_index
        raw = self.message(torch.cat([h[source], h[destination], rbf], dim=-1))
        gate, candidate = raw.chunk(2, dim=-1)
        msg = torch.sigmoid(gate) * F.softplus(candidate)
        msg = msg * edge_weight.unsqueeze(-1)
        aggregate = segment_sum(msg, destination, h.shape[0])
        return self.norm(F.softplus(h + self.dropout(aggregate)))


class ControlledCGCNN(nn.Module):
    def __init__(self, *, hidden_dim=128, num_layers=4, num_rbf=32, cutoff=6.0, dropout=0.10):
        super().__init__()
        self.cutoff = float(cutoff)
        self.atom = AtomicEncoder(hidden_dim)
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.layers = nn.ModuleList([CGCNNLayer(hidden_dim, num_rbf, dropout) for _ in range(num_layers)])
        self.readout = BaselineReadout(hidden_dim, hidden_dim, dropout)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        h = self.atom(batch)
        distance, _ = edge_geometry(batch)
        radial = self.rbf(distance)
        edge_weight = self.rbf.cutoff_envelope(distance)
        for layer in self.layers:
            h = layer(h, batch["edge_index"], radial, edge_weight)
        return self.readout(segment_mean(h, batch["batch"], batch["lattice"].shape[0]))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int, dropout: float) -> None:
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, rbf, edge_weight):
        source, destination = edge_index
        filt = self.filter(rbf) * edge_weight.unsqueeze(-1)
        msg = filt * self.source(h[source])
        aggregate = segment_sum(msg, destination, h.shape[0])
        return self.norm(h + self.update(aggregate))


class ControlledSchNet(nn.Module):
    def __init__(self, *, hidden_dim=128, num_layers=4, num_rbf=48, cutoff=6.0, dropout=0.10):
        super().__init__()
        self.cutoff = float(cutoff)
        self.atom = AtomicEncoder(hidden_dim)
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.layers = nn.ModuleList([SchNetInteraction(hidden_dim, num_rbf, dropout) for _ in range(num_layers)])
        self.readout = BaselineReadout(hidden_dim, hidden_dim, dropout)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        h = self.atom(batch)
        distance, _ = edge_geometry(batch)
        radial = self.rbf(distance)
        edge_weight = self.rbf.cutoff_envelope(distance)
        for layer in self.layers:
            h = layer(h, batch["edge_index"], radial, edge_weight)
        return self.readout(segment_mean(h, batch["batch"], batch["lattice"].shape[0]))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AngleAwareLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf + 3, 2 * hidden_dim), nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, rbf, angle_features, edge_weight):
        source, destination = edge_index
        msg = self.message(torch.cat([h[source], h[destination], rbf, angle_features[destination]], dim=-1))
        msg = msg * edge_weight.unsqueeze(-1)
        aggregate = weighted_segment_mean(msg, destination, h.shape[0], edge_weight)
        update = self.update(torch.cat([aggregate, angle_features], dim=-1))
        return self.norm(h + update)


class ControlledALIGNN(nn.Module):
    """Internal angle-moment control; not the upstream ALIGNN implementation."""

    def __init__(self, *, hidden_dim=128, num_layers=4, num_rbf=48, cutoff=6.0, dropout=0.10):
        super().__init__()
        self.cutoff = float(cutoff)
        self.atom = AtomicEncoder(hidden_dim)
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.layers = nn.ModuleList([AngleAwareLayer(hidden_dim, num_rbf, dropout) for _ in range(num_layers)])
        self.readout = BaselineReadout(hidden_dim + 3, hidden_dim, dropout)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        h = self.atom(batch)
        distance, unit = edge_geometry(batch)
        radial = self.rbf(distance)
        edge_weight = self.rbf.cutoff_envelope(distance)
        destination = batch["edge_index"][1]
        angle_features = directional_invariants(unit, destination, h.shape[0], edge_weight)
        for layer in self.layers:
            h = layer(h, batch["edge_index"], radial, angle_features, edge_weight)
        n_graphs = batch["lattice"].shape[0]
        graph_h = segment_mean(h, batch["batch"], n_graphs)
        graph_angle = segment_mean(angle_features, batch["batch"], n_graphs)
        return self.readout(torch.cat([graph_h, graph_angle], dim=-1))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class StateAwareLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim + num_rbf + 3, 2 * hidden_dim), nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, state, batch_index, edge_index, rbf, angle_features, edge_weight):
        source, destination = edge_index
        edge_state = state[batch_index[destination]]
        msg = self.message(
            torch.cat([h[source], h[destination], edge_state, rbf, angle_features[destination]], dim=-1)
        )
        msg = msg * edge_weight.unsqueeze(-1)
        aggregate = weighted_segment_mean(msg, destination, h.shape[0], edge_weight)
        update = self.node_update(torch.cat([aggregate, state[batch_index], angle_features], dim=-1))
        return self.norm(h + update)


class ControlledM3GNet(nn.Module):
    """Internal state/three-body-moment control; not upstream M3GNet."""

    def __init__(self, *, hidden_dim=128, num_layers=4, num_rbf=48, cutoff=6.0, dropout=0.10):
        super().__init__()
        self.cutoff = float(cutoff)
        self.atom = AtomicEncoder(hidden_dim)
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.state_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([StateAwareLayer(hidden_dim, num_rbf, dropout) for _ in range(num_layers)])
        self.readout = BaselineReadout(2 * hidden_dim + 3, hidden_dim, dropout)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        h = self.atom(batch)
        state = self.state_encoder(batch["global_features"])
        distance, unit = edge_geometry(batch)
        radial = self.rbf(distance)
        edge_weight = self.rbf.cutoff_envelope(distance)
        destination = batch["edge_index"][1]
        angle_features = directional_invariants(unit, destination, h.shape[0], edge_weight)
        for layer in self.layers:
            h = layer(h, state, batch["batch"], batch["edge_index"], radial, angle_features, edge_weight)
        n_graphs = batch["lattice"].shape[0]
        graph_h = segment_mean(h, batch["batch"], n_graphs)
        graph_angle = segment_mean(angle_features, batch["batch"], n_graphs)
        return self.readout(torch.cat([graph_h, state, graph_angle], dim=-1))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(name: str, *, hidden_dim: int, num_layers: int, cutoff: float, dropout: float) -> nn.Module:
    name = name.lower()
    kwargs = {
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "cutoff": float(cutoff),
        "dropout": float(dropout),
    }
    if name == "cgcnn":
        return ControlledCGCNN(num_rbf=32, **kwargs)
    if name == "schnet":
        return ControlledSchNet(num_rbf=48, **kwargs)
    if name == "alignn":
        return ControlledALIGNN(num_rbf=48, **kwargs)
    if name == "m3gnet":
        return ControlledM3GNet(num_rbf=48, **kwargs)
    raise ValueError(f"unknown graph baseline: {name}")
