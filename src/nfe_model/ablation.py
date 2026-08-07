from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .model import PeriodicNFEModel, segment_max, segment_mean, segment_sum


class ScalarOnlyInteraction(nn.Module):
    """Capacity-matched scalar path of the full equivariant interaction.

    The module intentionally retains the same parameter shapes as the full
    interaction. Vector-producing parameters remain present but their outputs
    are not consumed, so the ablation removes directional/vector information
    without replacing the scalar message architecture or shrinking capacity.
    """

    def __init__(
        self,
        hidden_dim: int,
        vector_dim: int,
        num_rbf: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.vector_dim = int(vector_dim)
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim + 2 * vector_dim),
        )
        # Keep the full interaction's vector parameters for capacity matching.
        # They are deliberately unused in the scalar-only forward pass.
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
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source, destination = edge_index
        edge_features = torch.cat(
            [scalar[source], scalar[destination], rbf], dim=-1
        )
        messages = self.edge_mlp(edge_features)
        scalar_message = messages[:, : self.hidden_dim]
        if edge_weight is not None:
            scalar_message = scalar_message * edge_weight.unsqueeze(-1)
        aggregate = segment_sum(scalar_message, destination, scalar.shape[0])
        scalar = self.scalar_norm(scalar + self.dropout(aggregate))

        # The full interaction feeds vector norms into the same update MLP.
        # Setting that signal to zero removes vector information while keeping
        # the update architecture and parameter count unchanged.
        zero_vector_norm = scalar.new_zeros((scalar.shape[0], self.vector_dim))
        update = self.update_mlp(torch.cat([scalar, zero_vector_norm], dim=-1))
        scalar_update = update[:, : self.hidden_dim]
        return scalar + scalar_update


class AblationPeriodicNFEModel(PeriodicNFEModel):
    """PeriodicNFEModel variant that removes selected information branches.

    Representation ablations are capacity preserving: vector/global modules and
    the full readout dimensionality remain allocated, while the corresponding
    information channel is replaced by zeros. This avoids conflating an
    information ablation with a parameter-count/readout-capacity ablation.
    """

    def __init__(
        self,
        *,
        use_vector_features: bool = True,
        use_global_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_vector_features = bool(use_vector_features)
        self.use_global_features = bool(use_global_features)
        hidden_dim = int(self.config["hidden_dim"])
        vector_dim = int(self.config["vector_dim"])
        num_layers = int(self.config["num_layers"])
        num_rbf = int(self.config["num_rbf"])
        dropout = float(self.config["dropout"])

        if not self.use_vector_features:
            self.layers = nn.ModuleList(
                [
                    ScalarOnlyInteraction(hidden_dim, vector_dim, num_rbf, dropout)
                    for _ in range(num_layers)
                ]
            )

        # Keep the production readout dimensions for every representation
        # ablation. Disabled branches contribute zero tensors of the same size.
        readout_input = 3 * hidden_dim + vector_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_input, 2 * hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.config["use_vector_features"] = self.use_vector_features
        self.config["use_global_features"] = self.use_global_features
        self.config["capacity_preserving_ablation"] = True

    def encode(
        self,
        batch_data: dict[str, Any],
        *,
        z_override: torch.Tensor | None = None,
        frac_pos_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = batch_data["z"] if z_override is None else z_override
        frac_pos = (
            batch_data["frac_pos"]
            if frac_pos_override is None
            else frac_pos_override
        )
        z = torch.clamp(z, 0, self.max_atomic_number)
        descriptors = batch_data["atom_features"]
        if z_override is not None:
            descriptors = descriptors * (z > 0).unsqueeze(-1)
        scalar = self.atom_embedding(z) + self.element_encoder(descriptors)
        vector = scalar.new_zeros(
            (scalar.shape[0], 3, int(self.config["vector_dim"]))
        )
        _, distance, unit = self.edge_geometry(
            frac_pos,
            batch_data["lattice"],
            batch_data["batch"],
            batch_data["edge_index"],
            batch_data["edge_shift"],
        )
        radial = self.rbf(distance)
        if self.use_vector_features:
            for layer in self.layers:
                scalar, vector = layer(
                    scalar,
                    vector,
                    batch_data["edge_index"],
                    unit,
                    radial,
                )
        else:
            for layer in self.layers:
                scalar = layer(scalar, batch_data["edge_index"], radial)

        n_graphs = int(batch_data["lattice"].shape[0])
        graph_index = batch_data["batch"]
        gate = torch.sigmoid(self.pool_gate(scalar))
        gated_sum = segment_sum(gate * scalar, graph_index, n_graphs)
        gate_sum = segment_sum(gate, graph_index, n_graphs).clamp_min(1e-6)
        attention_pool = gated_sum / gate_sum
        max_pool = segment_max(scalar, graph_index, n_graphs)

        if self.use_vector_features:
            vector_norm = torch.sqrt(torch.sum(vector * vector, dim=1) + 1e-8)
            vector_pool = segment_mean(vector_norm, graph_index, n_graphs)
        else:
            vector_pool = scalar.new_zeros((n_graphs, int(self.config["vector_dim"])))

        if self.use_global_features:
            global_pool = self.global_encoder(batch_data["global_features"])
        else:
            global_pool = scalar.new_zeros((n_graphs, int(self.config["hidden_dim"])))

        graph_embedding = self.readout(
            torch.cat([attention_pool, max_pool, vector_pool, global_pool], dim=-1)
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
        if self.use_vector_features:
            denoise_vector = self.denoise_head(vector).squeeze(-1)
        else:
            # Keep denoise_head parameters allocated but remove its information
            # path and objective for the no-vector ablation.
            denoise_vector = scalar.new_zeros((scalar.shape[0], 3))
        return {
            "class_logits": self.classifier(embedding),
            "regression_mean": mean,
            "regression_log_variance": torch.clamp(log_variance, -8.0, 5.0),
            "masked_atom_logits": self.masked_atom_head(scalar),
            "denoise_vector": denoise_vector,
            "embedding": embedding,
        }
