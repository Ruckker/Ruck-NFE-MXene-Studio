# ==============================================================================
# 中文概述：表面感知模板条件流：分别建模内核、端基和晶格。
# English overview: Surface-aware template flow that treats the core, terminations, and lattice separately.
#
# 中文输入：带角色/锚点/层信息的噪声模板、时间和 NFE 条件。
# English inputs: Noisy templates with roles, anchors, layers, time, and NFE conditions.
# 中文输出：尊重二维周期性的坐标/晶格速度和端点重建。
# English outputs: Coordinate/lattice velocities and endpoint reconstruction respecting 2D periodicity.
#
# 关键约束 / Key invariants:
# - 二维/三维周期边界、分数坐标和晶格单位必须保持一致。
#   Periodic boundaries, fractional coordinates, and lattice units must stay consistent.
# - NFE 标签是从电子结构计算提取的伪标签；最终材料结论仍需 DFT/VASP 验证。
#   NFE labels are electronic-structure-derived pseudo-labels; final claims still require DFT/VASP.
# - 主要接口 / Main APIs: periodic_xy_knn_edges, surface_coordinate_length_scale, SurfaceAwareTemplateFlow
#
# Author: Ruck
# Generated: 2026-07-29 22:30:47 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import EquivariantInteraction, GaussianRBF, segment_mean
from .time_embedding import FourierTimeEmbedding


# 中文：顶层接口 `periodic_xy_knn_edges`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `periodic_xy_knn_edges`; review type hints and callers before extending it.
def periodic_xy_knn_edges(
    frac_pos: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sources, destinations, units, distances, graph_parts = [], [], [], [], []
    shifts = torch.tensor(
        [
            [i, j, 0.0]
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
        ],
        device=frac_pos.device,
        dtype=frac_pos.dtype,
    )
    for graph_index in range(lattice.shape[0]):
        node_indices = torch.where(batch == graph_index)[0]
        n_atoms = len(node_indices)
        if n_atoms <= 1:
            continue
        local_frac = frac_pos[node_indices]
        raw_delta = local_frac[:, None, :] - local_frac[None, :, :]
        candidates = raw_delta.unsqueeze(2) + shifts.view(1, 1, 9, 3)
        candidate_cartesian = torch.einsum(
            "sdki,ij->sdkj", candidates, lattice[graph_index]
        )
        candidate_distance = torch.sum(candidate_cartesian.square(), dim=-1)
        best_image = candidate_distance.argmin(dim=2)
        gather_index = best_image[..., None, None].expand(-1, -1, 1, 3)
        cartesian = torch.gather(
            candidate_cartesian, 2, gather_index
        ).squeeze(2)
        distance = torch.linalg.vector_norm(cartesian, dim=-1)
        distance.fill_diagonal_(torch.inf)
        neighbors = min(max_neighbors, n_atoms - 1)
        nearest_distance, local_source = torch.topk(
            distance, k=neighbors, dim=0, largest=False, sorted=False
        )
        local_destination = (
            torch.arange(n_atoms, device=frac_pos.device)
            .unsqueeze(0)
            .expand(neighbors, -1)
        )
        edge_cartesian = cartesian[local_source, local_destination]
        sources.append(node_indices[local_source.reshape(-1)])
        destinations.append(node_indices[local_destination.reshape(-1)])
        flat_distance = nearest_distance.reshape(-1).clamp_min(1e-7)
        distances.append(flat_distance)
        units.append(edge_cartesian.reshape(-1, 3) / flat_distance.unsqueeze(-1))
        graph_parts.append(
            torch.full(
                (flat_distance.numel(),),
                graph_index,
                device=frac_pos.device,
                dtype=torch.long,
            )
        )
    if not sources:
        raise ValueError("surface generator requires at least two atoms per structure")
    return (
        torch.stack([torch.cat(sources), torch.cat(destinations)], dim=0),
        torch.cat(units),
        torch.cat(distances),
        torch.cat(graph_parts),
    )


# 中文：顶层接口 `surface_coordinate_length_scale`；先阅读类型标注与调用方再扩展实现。
# English: Top-level function `surface_coordinate_length_scale`; review type hints and callers before extending it.
def surface_coordinate_length_scale(
    lattice: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    counts = torch.bincount(batch, minlength=lattice.shape[0]).clamp_min(1)
    area = torch.linalg.vector_norm(
        torch.linalg.cross(lattice[:, 0], lattice[:, 1], dim=-1), dim=-1
    )
    return torch.sqrt(area.clamp_min(1e-6) / counts.to(area.dtype)).clamp(0.8, 4.0)


# 中文：顶层类 `SurfaceAwareTemplateFlow`；先阅读类型标注与调用方再扩展实现。
# English: Top-level class `SurfaceAwareTemplateFlow`; review type hints and callers before extending it.
class SurfaceAwareTemplateFlow(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        vector_dim: int = 64,
        num_layers: int = 6,
        num_rbf: int = 64,
        cutoff: float = 12.0,
        max_neighbors: int = 24,
        dropout: float = 0.10,
        max_atomic_number: int = 118,
        element_features: int = 14,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.config = {
            "hidden_dim": hidden_dim,
            "vector_dim": vector_dim,
            "num_layers": num_layers,
            "num_rbf": num_rbf,
            "cutoff": cutoff,
            "max_neighbors": max_neighbors,
            "dropout": dropout,
            "max_atomic_number": max_atomic_number,
            "element_features": element_features,
            "condition_dim": condition_dim,
        }
        self.max_atomic_number = max_atomic_number
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.vector_dim = vector_dim
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.element_encoder = nn.Sequential(
            nn.Linear(element_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_fourier = FourierTimeEmbedding(condition_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.label_embedding = nn.Embedding(4, hidden_dim)
        self.score_encoder = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim),
        )
        self.lattice_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.count_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.side_embedding = nn.Embedding(4, hidden_dim)
        self.group_embedding = nn.Embedding(6, hidden_dim)
        self.adsorption_embedding = nn.Embedding(6, hidden_dim)
        self.layer_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.template_vector_encoder = nn.Sequential(
            nn.Linear(num_rbf, vector_dim),
            nn.SiLU(),
            nn.Linear(vector_dim, vector_dim),
        )
        self.layers = nn.ModuleList(
            [
                EquivariantInteraction(
                    hidden_dim, vector_dim, num_rbf, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.coord_head = nn.Linear(vector_dim, 1, bias=False)
        self.lattice_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 6),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.atom_embedding.weight, std=0.02)
        nn.init.normal_(self.label_embedding.weight, std=0.02)
        nn.init.normal_(self.side_embedding.weight, std=0.02)
        nn.init.normal_(self.group_embedding.weight, std=0.02)
        nn.init.normal_(self.adsorption_embedding.weight, std=0.02)
        nn.init.zeros_(self.lattice_head[-1].bias)

    def condition(
        self,
        time: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        lattice_state: torch.Tensor,
        atom_counts: torch.Tensor,
    ) -> torch.Tensor:
        label_index = torch.where(
            labels >= 0, labels.clamp(0, 2), torch.full_like(labels, 3)
        )
        return (
            self.time_encoder(self.time_fourier(time))
            + self.label_embedding(label_index)
            + self.score_encoder(scores.unsqueeze(-1))
            + self.lattice_encoder(lattice_state)
            + self.count_encoder(torch.log1p(atom_counts.float()).unsqueeze(-1))
        )

    def forward(
        self,
        batch_data: dict[str, torch.Tensor],
        frac_pos: torch.Tensor,
        lattice_state: torch.Tensor,
        lattice: torch.Tensor,
        time: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z = batch_data["z"].clamp(0, self.max_atomic_number)
        graph_index = batch_data["batch"]
        atom_counts = torch.bincount(
            graph_index, minlength=lattice.shape[0]
        )
        condition = self.condition(
            time, labels, scores, lattice_state, atom_counts
        )
        side = batch_data["surface_side"].clamp(-1, 1) + 1
        group_type = batch_data["group_type"].clamp(0, 4)
        adsorption = batch_data["adsorption_coordination"].clamp(0, 5)
        scalar = (
            self.atom_embedding(z)
            + self.element_encoder(batch_data["atom_features"])
            + condition[graph_index]
            + self.side_embedding(side)
            + self.group_embedding(group_type)
            + self.adsorption_embedding(adsorption)
            + self.layer_encoder(batch_data["layer_position"].unsqueeze(-1))
        )

        template_delta = batch_data["template_frac"] - frac_pos
        template_delta = template_delta.clone()
        template_delta[:, :2] = (
            template_delta[:, :2] + 0.5
        ) % 1.0 - 0.5
        template_cartesian = torch.einsum(
            "ni,nij->nj", template_delta, lattice[graph_index]
        )
        template_distance = torch.linalg.vector_norm(
            template_cartesian, dim=-1
        ).clamp_min(1e-7)
        template_unit = template_cartesian / template_distance.unsqueeze(-1)
        template_radial = self.rbf(template_distance.clamp_max(self.cutoff))
        template_channels = self.template_vector_encoder(template_radial)
        vector = template_unit.unsqueeze(-1) * template_channels.unsqueeze(1)

        edge_index, unit, distance, _ = periodic_xy_knn_edges(
            frac_pos, lattice, graph_index, self.max_neighbors
        )
        radial = self.rbf(distance)
        normalized_distance = torch.clamp(distance / self.cutoff, 0.0, 1.0)
        envelope = 0.5 * (
            torch.cos(math.pi * normalized_distance) + 1.0
        )
        envelope = envelope * (distance < self.cutoff)
        for layer in self.layers:
            scalar, vector = layer(
                scalar,
                vector,
                edge_index,
                unit,
                radial,
                edge_weight=envelope,
            )
        graph_scalar = segment_mean(
            scalar, graph_index, lattice.shape[0]
        )
        lattice_velocity = self.lattice_head(
            torch.cat(
                [
                    graph_scalar,
                    condition,
                    self.lattice_encoder(lattice_state),
                ],
                dim=-1,
            )
        )
        return {
            "coordinate_velocity_cart": self.coord_head(vector).squeeze(-1),
            "lattice_velocity": lattice_velocity,
            "node_embedding": scalar,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
