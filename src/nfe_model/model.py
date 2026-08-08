# ==============================================================================
# 中文概述：NFE 三分类与多物性回归使用的周期等变图神经网络。
# English overview: Periodic equivariant graph neural network for NFE tri-classification and multi-property regression.
#
# 中文输入：原子特征、周期边、距离/方向、晶胞不变量与批次索引。
# English inputs: Atomic features, periodic edges, distances/directions, cell invariants, and batch indices.
# 中文输出：low/medium/high logits、异方差回归、图嵌入与不确定性信息。
# English outputs: Low/medium/high logits, heteroscedastic regression, graph embeddings, and uncertainty signals.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: segment_sum, segment_mean, segment_max, GaussianRBF, EquivariantInteraction, PeriodicNFEModel, enable_mc_dropout, heteroscedastic_loss
#
# Author: Ruck
# Generated: 2026-07-29 19:06:31 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_sum(
    values: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    output = values.new_zeros((dim_size,) + values.shape[1:])
    output.index_add_(0, index, values)
    return output


def segment_mean(
    values: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    output = segment_sum(values, index, dim_size)
    count = values.new_zeros(dim_size)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return output / count.clamp_min(1.0).view((-1,) + (1,) * (values.ndim - 1))


def segment_max(
    values: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    output = values.new_full((dim_size,) + values.shape[1:], -torch.inf)
    expanded = index.view((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
    output.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.nan_to_num(output, neginf=0.0)


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        if int(num_rbf) <= 0:
            raise ValueError("num_rbf must be positive")
        if float(cutoff) <= 0:
            raise ValueError("cutoff must be positive")
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        spacing = float(centers[1] - centers[0]) if num_rbf > 1 else cutoff
        self.gamma = 1.0 / max(spacing * spacing, 1e-8)
        self.cutoff = float(cutoff)

    def cutoff_envelope(self, distance: torch.Tensor) -> torch.Tensor:
        """Cosine edge gate that is exactly zero at and beyond the cutoff."""
        x = torch.clamp(distance / self.cutoff, 0.0, 1.0)
        envelope = 0.5 * (torch.cos(math.pi * x) + 1.0)
        return envelope * (distance < self.cutoff).to(envelope.dtype)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(
            -self.gamma * (distance.unsqueeze(-1) - self.centers) ** 2
        )
        return rbf * self.cutoff_envelope(distance).unsqueeze(-1)


class EquivariantInteraction(nn.Module):
    """Lightweight PaiNN-style scalar/vector interaction.

    Scalar channels are invariant. Vector channels transform equivariantly
    because they are formed only from existing vectors and unit bond vectors
    multiplied by invariant gates.
    """

    def __init__(
        self,
        hidden_dim: int,
        vector_dim: int,
        num_rbf: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim + 2 * vector_dim),
        )
        self.vector_source = nn.Linear(vector_dim, vector_dim, bias=False)
        self.vector_update = nn.Linear(vector_dim, vector_dim, bias=False)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + vector_dim, 2 * hidden_dim + vector_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim + vector_dim, hidden_dim + vector_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        unit: torch.Tensor,
        rbf: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        edge_features = torch.cat(
            [scalar[source], scalar[destination], rbf], dim=-1
        )
        messages = self.edge_mlp(edge_features)
        scalar_message, source_gate, radial_gate = torch.split(
            messages,
            [self.hidden_dim, self.vector_dim, self.vector_dim],
            dim=-1,
        )
        if edge_weight is not None:
            scalar_message = scalar_message * edge_weight.unsqueeze(-1)
            source_gate = source_gate * edge_weight.unsqueeze(-1)
            radial_gate = radial_gate * edge_weight.unsqueeze(-1)
        source_vector = self.vector_source(vector[source])
        vector_message = (
            source_gate.unsqueeze(1) * source_vector
            + radial_gate.unsqueeze(1) * unit.unsqueeze(-1)
        )
        n_nodes = scalar.shape[0]
        scalar_aggregate = segment_sum(scalar_message, destination, n_nodes)
        vector_aggregate = segment_sum(vector_message, destination, n_nodes)

        scalar = self.scalar_norm(scalar + self.dropout(scalar_aggregate))
        vector = vector + vector_aggregate
        mixed_vector = self.vector_update(vector)
        vector_norm = torch.sqrt(
            torch.sum(mixed_vector * mixed_vector, dim=1) + 1e-8
        )
        update = self.update_mlp(torch.cat([scalar, vector_norm], dim=-1))
        scalar_update, vector_gate = torch.split(
            update, [self.hidden_dim, self.vector_dim], dim=-1
        )
        scalar = scalar + scalar_update
        vector = vector + torch.sigmoid(vector_gate).unsqueeze(1) * mixed_vector
        return scalar, vector


class PeriodicNFEModel(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        vector_dim: int = 64,
        num_layers: int = 6,
        num_rbf: int = 48,
        cutoff: float = 6.0,
        dropout: float = 0.12,
        max_atomic_number: int = 118,
        element_features: int = 14,
        global_features: int = 11,
        num_regression_targets: int = 10,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if int(max_atomic_number) <= 0:
            raise ValueError("max_atomic_number must be positive")
        self.config = {
            "hidden_dim": hidden_dim,
            "vector_dim": vector_dim,
            "num_layers": num_layers,
            "num_rbf": num_rbf,
            "cutoff": cutoff,
            "dropout": dropout,
            "max_atomic_number": max_atomic_number,
            "element_features": element_features,
            "global_features": global_features,
            "num_regression_targets": num_regression_targets,
            "num_classes": num_classes,
        }
        self.cutoff = float(cutoff)
        self.max_atomic_number = int(max_atomic_number)
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.element_encoder = nn.Sequential(
            nn.Linear(element_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.layers = nn.ModuleList(
            [
                EquivariantInteraction(
                    hidden_dim, vector_dim, num_rbf, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_features, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.pool_gate = nn.Linear(hidden_dim, 1)
        readout_input = 2 * hidden_dim + vector_dim + hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_input, 2 * hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.regression_head = nn.Linear(hidden_dim, 2 * num_regression_targets)
        self.masked_atom_head = nn.Linear(hidden_dim, max_atomic_number + 1)
        self.denoise_head = nn.Linear(vector_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.atom_embedding.weight, std=0.02)
        with torch.no_grad():
            self.atom_embedding.weight[0].zero_()
        nn.init.zeros_(self.regression_head.bias)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def edge_geometry(
        frac_pos: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        edge_lattice = lattice[batch[destination]]
        delta_fractional = (
            frac_pos[source] + edge_shift - frac_pos[destination]
        )
        delta_cartesian = torch.einsum(
            "ei,eij->ej", delta_fractional, edge_lattice
        )
        distance = torch.linalg.vector_norm(delta_cartesian, dim=-1).clamp_min(1e-8)
        unit = delta_cartesian / distance.unsqueeze(-1)
        return delta_cartesian, distance, unit

    def encode(
        self,
        batch_data: dict[str, Any],
        *,
        z_override: torch.Tensor | None = None,
        frac_pos_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = batch_data["z"] if z_override is None else z_override
        if torch.any(z < 0) or torch.any(z > self.max_atomic_number):
            minimum = int(z.min().detach().cpu()) if z.numel() else 0
            maximum = int(z.max().detach().cpu()) if z.numel() else 0
            raise ValueError(
                "atomic number outside model vocabulary: "
                f"observed=[{minimum}, {maximum}] allowed=[0, {self.max_atomic_number}]"
            )
        frac_pos = (
            batch_data["frac_pos"]
            if frac_pos_override is None
            else frac_pos_override
        )
        descriptors = batch_data["atom_features"]
        if z_override is not None:
            descriptors = descriptors * (z > 0).unsqueeze(-1)
        scalar = self.atom_embedding(z) + self.element_encoder(descriptors)
        vector = scalar.new_zeros(
            (scalar.shape[0], 3, self.config["vector_dim"])
        )
        _, distance, unit = self.edge_geometry(
            frac_pos,
            batch_data["lattice"],
            batch_data["batch"],
            batch_data["edge_index"],
            batch_data["edge_shift"],
        )
        radial = self.rbf(distance)
        edge_weight = self.rbf.cutoff_envelope(distance)
        for layer in self.layers:
            scalar, vector = layer(
                scalar,
                vector,
                batch_data["edge_index"],
                unit,
                radial,
                edge_weight=edge_weight,
            )

        n_graphs = int(batch_data["lattice"].shape[0])
        graph_index = batch_data["batch"]
        gate = torch.sigmoid(self.pool_gate(scalar))
        gated_sum = segment_sum(gate * scalar, graph_index, n_graphs)
        gate_sum = segment_sum(gate, graph_index, n_graphs).clamp_min(1e-6)
        attention_pool = gated_sum / gate_sum
        max_pool = segment_max(scalar, graph_index, n_graphs)
        vector_norm = torch.sqrt(torch.sum(vector * vector, dim=1) + 1e-8)
        vector_pool = segment_mean(vector_norm, graph_index, n_graphs)
        global_encoded = self.global_encoder(batch_data["global_features"])
        graph_embedding = self.readout(
            torch.cat(
                [attention_pool, max_pool, vector_pool, global_encoded], dim=-1
            )
        )
        return scalar, vector, graph_embedding

    def forward(
        self,
        batch_data: dict[str, Any],
        *,
        z_override: torch.Tensor | None = None,
        frac_pos_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scalar, vector, embedding = self.encode(
            batch_data,
            z_override=z_override,
            frac_pos_override=frac_pos_override,
        )
        regression = self.regression_head(embedding)
        mean, log_variance = regression.chunk(2, dim=-1)
        return {
            "class_logits": self.classifier(embedding),
            "regression_mean": mean,
            "regression_log_variance": torch.clamp(log_variance, -8.0, 5.0),
            "masked_atom_logits": self.masked_atom_head(scalar),
            "denoise_vector": self.denoise_head(vector).squeeze(-1),
            "embedding": embedding,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def heteroscedastic_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    squared = (mean - target) ** 2
    loss = 0.5 * torch.exp(-log_variance) * squared + 0.5 * log_variance
    if target_weights is not None:
        loss = loss * target_weights.view(1, -1)
    effective_mask = mask.to(loss.dtype)
    if sample_weights is not None:
        effective_mask = effective_mask * sample_weights.view(-1, 1)
    denominator = effective_mask.sum()
    if float(denominator.detach()) <= 0:
        return mean.sum() * 0.0
    return torch.sum(loss * effective_mask) / denominator
