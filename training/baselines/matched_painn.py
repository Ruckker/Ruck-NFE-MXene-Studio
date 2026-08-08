from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nfe_model.model import (
    EquivariantInteraction,
    GaussianRBF,
    PeriodicNFEModel,
    segment_max,
    segment_mean,
    segment_sum,
)


class MatchedPaiNNBaseline(nn.Module):
    """PaiNN-style Ruck-NFE backbone with only class + NFE-score supervision.

    It intentionally excludes global slab features, auxiliary electronic-property
    heads, masked-atom prediction and coordinate denoising so architecture-track
    supervision matches the controlled graph baselines.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 192,
        num_layers: int = 6,
        num_rbf: int = 48,
        cutoff: float = 6.0,
        dropout: float = 0.12,
        max_atomic_number: int = 118,
        element_features: int = 14,
    ) -> None:
        super().__init__()
        vector_dim = max(16, hidden_dim // 3)
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
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
                EquivariantInteraction(hidden_dim, vector_dim, num_rbf, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim + vector_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.score_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.atom_embedding.weight, std=0.02)
        with torch.no_grad():
            self.atom_embedding.weight[0].zero_()
        nn.init.zeros_(self.classifier.bias)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        z = batch["z"]
        if torch.any(z < 0) or torch.any(z > self.max_atomic_number):
            minimum = int(z.min().detach().cpu()) if z.numel() else 0
            maximum = int(z.max().detach().cpu()) if z.numel() else 0
            raise ValueError(
                "atomic number outside matched PaiNN vocabulary: "
                f"observed=[{minimum}, {maximum}] allowed=[0, {self.max_atomic_number}]"
            )
        scalar = self.atom_embedding(z) + self.element_encoder(batch["atom_features"])
        vector = scalar.new_zeros((scalar.shape[0], 3, self.vector_dim))
        _, distance, unit = PeriodicNFEModel.edge_geometry(
            batch["frac_pos"],
            batch["lattice"],
            batch["batch"],
            batch["edge_index"],
            batch["edge_shift"],
        )
        radial = self.rbf(distance)
        edge_weight = self.rbf.cutoff_envelope(distance)
        for layer in self.layers:
            scalar, vector = layer(
                scalar,
                vector,
                batch["edge_index"],
                unit,
                radial,
                edge_weight=edge_weight,
            )

        n_graphs = int(batch["lattice"].shape[0])
        graph_index = batch["batch"]
        gate = torch.sigmoid(self.pool_gate(scalar))
        attention = segment_sum(gate * scalar, graph_index, n_graphs) / segment_sum(
            gate, graph_index, n_graphs
        ).clamp_min(1e-6)
        maximum = segment_max(scalar, graph_index, n_graphs)
        vector_norm = torch.sqrt(torch.sum(vector * vector, dim=1) + 1e-8)
        vector_pool = segment_mean(vector_norm, graph_index, n_graphs)
        embedding = self.readout(torch.cat([attention, maximum, vector_pool], dim=-1))
        return {
            "class_logits": self.classifier(embedding),
            "score": self.score_head(embedding).squeeze(-1),
            "embedding": embedding,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
