from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pymatgen.core import Element


def _segment_mean(values: torch.Tensor, index: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = values.new_zeros((n_graphs,) + values.shape[1:])
    out.index_add_(0, index, values)
    count = values.new_zeros(n_graphs)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return out / count.clamp_min(1.0).view((-1,) + (1,) * (values.ndim - 1))


def _edge_geometry(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    source, destination = batch["edge_index"]
    edge_lattice = batch["lattice"][batch["batch"][destination]]
    delta_fractional = (
        batch["frac_pos"][source]
        + batch["edge_shift"]
        - batch["frac_pos"][destination]
    )
    delta_cartesian = torch.einsum("ei,eij->ej", delta_fractional, edge_lattice)
    distance = torch.linalg.vector_norm(delta_cartesian, dim=-1).clamp_min(1e-8)
    return delta_cartesian, distance


class _DualHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, 4)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projection(embedding)
        return {"class_logits": raw[:, :3], "score": raw[:, 3]}


class OfficialSchNetPack(nn.Module):
    """Official SchNetPack SchNet representation on the common v2 periodic edge list."""

    def __init__(self, hidden_dim: int, num_layers: int, cutoff: float) -> None:
        super().__init__()
        try:
            import schnetpack as spk
            from schnetpack.nn.radial import GaussianRBF
            from schnetpack.nn.cutoff import CosineCutoff
            from schnetpack.representation import SchNet
        except ImportError as exc:
            raise RuntimeError(
                "SchNetPack backend requires an isolated environment with schnetpack==2.2.0"
            ) from exc
        self.properties = spk.properties
        radial = GaussianRBF(n_rbf=64, cutoff=float(cutoff))
        cutoff_fn = CosineCutoff(float(cutoff))
        self.representation = SchNet(
            n_atom_basis=int(hidden_dim),
            n_interactions=int(num_layers),
            radial_basis=radial,
            cutoff_fn=cutoff_fn,
            nuclear_embedding=nn.Embedding(119, int(hidden_dim)),
        )
        self.head = _DualHead(int(hidden_dim))

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        source, destination = batch["edge_index"]
        delta_cartesian, _ = _edge_geometry(batch)
        properties = self.properties
        inputs = {
            properties.Z: batch["z"],
            properties.Rij: delta_cartesian,
            properties.idx_i: destination,
            properties.idx_j: source,
        }
        result = self.representation(inputs)
        scalar = result.get("scalar_representation")
        if scalar is None:
            property_key = getattr(
                properties, "scalar_representation", "scalar_representation"
            )
            scalar = result.get(property_key)
        if scalar is None:
            raise RuntimeError("SchNetPack did not return scalar_representation")
        graph = _segment_mean(scalar, batch["batch"], batch["lattice"].shape[0])
        return self.head(graph)


class OfficialMatGLM3GNet(nn.Module):
    """Official MatGL M3GNet consuming the common v2 periodic edge list."""

    def __init__(
        self,
        element_types: list[str],
        hidden_dim: int,
        num_layers: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        try:
            from matgl.models import M3GNet
        except ImportError as exc:
            raise RuntimeError(
                "MatGL backend requires an isolated environment with matgl==4.0.3"
            ) from exc
        kwargs = {
            "element_types": tuple(element_types),
            "dim_node_embedding": int(hidden_dim),
            "nblocks": int(num_layers),
            "cutoff": float(cutoff),
            "threebody_cutoff": min(float(cutoff), 4.0),
            "task_type": "regression",
            "is_intensive": True,
            "ntargets": 4,
        }
        signature = inspect.signature(M3GNet.__init__)
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        self.model = M3GNet(**kwargs)
        mapping = torch.full((119,), -1, dtype=torch.long)
        for index, symbol in enumerate(element_types):
            atomic_number = int(Element(symbol).Z)
            if atomic_number < len(mapping):
                mapping[atomic_number] = index
        self.register_buffer("z_to_type", mapping, persistent=True)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        try:
            from torch_geometric.data import Batch, Data
        except ImportError as exc:
            raise RuntimeError("MatGL 4.x requires torch-geometric") from exc

        graph_count = int(batch["lattice"].shape[0])
        source_neighbor, destination_center = batch["edge_index"]
        graph_of_edge = batch["batch"][destination_center]
        graphs = []
        for graph_index in range(graph_count):
            node_mask = batch["batch"] == graph_index
            node_indices = torch.nonzero(node_mask, as_tuple=False).flatten()
            if not len(node_indices):
                raise RuntimeError(f"empty graph {graph_index} in MatGL adapter")
            start = int(node_indices[0])
            edge_mask = graph_of_edge == graph_index
            center = destination_center[edge_mask] - start
            neighbor = source_neighbor[edge_mask] - start
            edge_index = torch.stack([center, neighbor], dim=0)
            lattice = batch["lattice"][graph_index]
            frac = batch["frac_pos"][node_indices]
            pos = torch.einsum("ni,ij->nj", frac, lattice)
            offshift = torch.einsum(
                "ei,ij->ej", batch["edge_shift"][edge_mask], lattice
            )
            atomic_numbers = batch["z"][node_indices]
            node_type = self.z_to_type[atomic_numbers]
            if torch.any(node_type < 0):
                unknown = sorted(
                    set(
                        int(value)
                        for value in atomic_numbers[node_type < 0].detach().cpu().tolist()
                    )
                )
                raise ValueError(
                    f"MatGL element vocabulary does not contain atomic numbers {unknown}"
                )
            graphs.append(
                Data(
                    node_type=node_type,
                    pos=pos,
                    edge_index=edge_index,
                    pbc_offshift=offshift,
                )
            )
        graph_batch = Batch.from_data_list(graphs).to(batch["z"].device)
        out = self.model(graph_batch)
        if isinstance(out, dict):
            out = out.get("output", out.get("pred", out.get("energy")))
        out = torch.as_tensor(out, device=batch["z"].device)
        if out.ndim == 1:
            out = out.view(-1, 4)
        if out.shape[-1] != 4:
            raise RuntimeError(f"MatGL M3GNet expected 4 outputs, got {tuple(out.shape)}")
        return {"class_logits": out[:, :3], "score": out[:, 3]}


class OfficialALIGNN(nn.Module):
    """Official ALIGNN message-passing backbone on the common v2 graph plus real line graphs."""

    def __init__(self, hidden_dim: int, num_layers: int, cutoff: float, max_neighbors: int) -> None:
        super().__init__()
        del cutoff, max_neighbors
        try:
            import dgl
            from alignn.graphs import compute_bond_cosines
            from alignn.models.alignn import ALIGNN, ALIGNNConfig
        except ImportError as exc:
            raise RuntimeError(
                "ALIGNN backend requires an isolated environment with alignn==2026.5.20"
            ) from exc
        self.dgl = dgl
        self.compute_bond_cosines = compute_bond_cosines
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=int(num_layers),
            gcn_layers=int(num_layers),
            atom_input_features=14,
            hidden_features=int(hidden_dim),
            output_features=4,
            classification=False,
        )
        self.model = ALIGNN(config)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        graph_count = int(batch["lattice"].shape[0])
        source, destination = batch["edge_index"]
        graph_of_edge = batch["batch"][destination]
        delta_cartesian, _ = _edge_geometry(batch)
        graphs = []
        line_graphs = []
        lattices = []
        for graph_index in range(graph_count):
            node_indices = torch.nonzero(
                batch["batch"] == graph_index, as_tuple=False
            ).flatten()
            if not len(node_indices):
                raise RuntimeError(f"empty graph {graph_index} in ALIGNN adapter")
            start = int(node_indices[0])
            edge_mask = graph_of_edge == graph_index
            local_source = source[edge_mask] - start
            local_destination = destination[edge_mask] - start
            graph = self.dgl.graph(
                (local_source, local_destination),
                num_nodes=len(node_indices),
                device=batch["z"].device,
            )
            graph.ndata["atom_features"] = batch["atom_features"][node_indices]
            graph.edata["r"] = -delta_cartesian[edge_mask]
            line_graph = graph.line_graph(shared=True)
            line_graph.apply_edges(self.compute_bond_cosines)
            graphs.append(graph)
            line_graphs.append(line_graph)
            lattices.append(batch["lattice"][graph_index])
        graph_batch = self.dgl.batch(graphs)
        line_batch = self.dgl.batch(line_graphs)
        lattice_batch = torch.stack(lattices)
        out = self.model((graph_batch, line_batch, lattice_batch))
        if out.ndim == 1:
            out = out.unsqueeze(0)
        return {"class_logits": out[:, :3], "score": out[:, 3]}


class OfficialCGCNN(nn.Module):
    """Original txie-93 CGCNN network on common v2 nodes/edges and project elemental features."""

    def __init__(
        self,
        cgcnn_repo: str | Path,
        hidden_dim: int,
        num_layers: int,
        cutoff: float,
        neighbor_slots: int,
        element_feature_dim: int = 14,
    ) -> None:
        super().__init__()
        repo = Path(cgcnn_repo).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from cgcnn.data import GaussianDistance
            from cgcnn.model import CrystalGraphConvNet
        except ImportError as exc:
            raise RuntimeError(
                "CGCNN backend requires a checkout of https://github.com/txie-93/cgcnn"
            ) from exc
        gdf = GaussianDistance(dmin=0, dmax=float(cutoff), step=0.2)
        filter_values = torch.as_tensor(gdf.filter, dtype=torch.float32)
        self.register_buffer("gaussian_filter", filter_values, persistent=True)
        self.gaussian_var = float(gdf.var)
        nbr_fea_len = int(filter_values.numel())
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=int(element_feature_dim),
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=int(hidden_dim),
            n_conv=int(num_layers),
            h_fea_len=int(hidden_dim),
            n_h=1,
            classification=False,
        )
        self.model.fc_out = nn.Linear(self.model.fc_out.in_features, 4)
        self.cutoff = float(cutoff)
        self.neighbor_slots = int(neighbor_slots)
        if self.neighbor_slots <= 0:
            raise ValueError("CGCNN neighbor_slots must be positive")

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        source, destination = batch["edge_index"]
        _, distance = _edge_geometry(batch)
        n_nodes = int(batch["z"].shape[0])
        degree = torch.bincount(destination, minlength=n_nodes)
        if int(degree.max().item()) > self.neighbor_slots:
            raise RuntimeError(
                "CGCNN validation/test graph exceeds the train-derived fixed neighbor slots: "
                f"observed={int(degree.max().item())} train_slots={self.neighbor_slots}. "
                "Do not truncate the common v2 edge list; report this adapter/OOD incompatibility."
            )
        nbr_index = torch.zeros(
            (n_nodes, self.neighbor_slots), dtype=torch.long, device=batch["z"].device
        )
        nbr_distance = torch.full(
            (n_nodes, self.neighbor_slots),
            self.cutoff + 1.0,
            dtype=distance.dtype,
            device=distance.device,
        )
        for node in range(n_nodes):
            edge_ids = torch.nonzero(destination == node, as_tuple=False).flatten()
            if len(edge_ids):
                order = edge_ids[torch.argsort(distance[edge_ids])]
                count = len(order)
                nbr_index[node, :count] = source[order]
                nbr_distance[node, :count] = distance[order]
        filters = self.gaussian_filter.to(nbr_distance)
        nbr_fea = torch.exp(
            -((nbr_distance.unsqueeze(-1) - filters) ** 2) / (self.gaussian_var**2)
        )
        crystal_atom_idx = [
            torch.nonzero(batch["batch"] == graph_index, as_tuple=False).flatten()
            for graph_index in range(int(batch["lattice"].shape[0]))
        ]
        out = self.model(
            batch["atom_features"], nbr_fea, nbr_index, crystal_atom_idx
        )
        if out.ndim == 1:
            out = out.unsqueeze(0)
        return {"class_logits": out[:, :3], "score": out[:, 3]}


def build_official_backend(
    name: str,
    *,
    element_types: list[str],
    hidden_dim: int,
    num_layers: int,
    cutoff: float,
    max_neighbors: int,
    cgcnn_repo: str | None = None,
    cgcnn_atom_init: str | None = None,
    cgcnn_neighbor_slots: int | None = None,
) -> nn.Module:
    del cgcnn_atom_init
    if name == "schnet_official":
        return OfficialSchNetPack(hidden_dim, num_layers, cutoff)
    if name == "m3gnet_official":
        return OfficialMatGLM3GNet(element_types, hidden_dim, num_layers, cutoff)
    if name == "alignn_official":
        return OfficialALIGNN(hidden_dim, num_layers, cutoff, max_neighbors)
    if name == "cgcnn_official":
        if not cgcnn_repo:
            raise ValueError("cgcnn_official requires --cgcnn-repo")
        return OfficialCGCNN(
            cgcnn_repo,
            hidden_dim,
            num_layers,
            cutoff,
            int(cgcnn_neighbor_slots or max_neighbors),
        )
    raise ValueError(f"unknown official backend: {name}")
