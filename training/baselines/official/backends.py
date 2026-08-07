from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pymatgen.core import Structure


def _segment_mean(values: torch.Tensor, index: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = values.new_zeros((n_graphs,) + values.shape[1:])
    out.index_add_(0, index, values)
    count = values.new_zeros(n_graphs)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return out / count.clamp_min(1.0).view((-1,) + (1,) * (values.ndim - 1))


class _DualHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, 4)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.projection(embedding)
        return {"class_logits": raw[:, :3], "score": raw[:, 3]}


class OfficialSchNetPack(nn.Module):
    """Official SchNetPack SchNet representation with the common NFE dual head."""

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
        )
        self.head = _DualHead(int(hidden_dim))

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        source, destination = batch["edge_index"]
        edge_lattice = batch["lattice"][batch["batch"][destination]]
        delta_fractional = (
            batch["frac_pos"][source]
            + batch["edge_shift"]
            - batch["frac_pos"][destination]
        )
        rij = torch.einsum("ei,eij->ej", delta_fractional, edge_lattice)
        p = self.properties
        inputs = {
            p.Z: batch["z"],
            p.Rij: rij,
            p.idx_i: destination,
            p.idx_j: source,
            p.idx_m: batch["batch"],
            p.n_atoms: torch.bincount(
                batch["batch"], minlength=batch["lattice"].shape[0]
            ),
        }
        result = self.representation(inputs)
        scalar = result.get("scalar_representation")
        if scalar is None:
            property_key = getattr(p, "scalar_representation", "scalar_representation")
            scalar = result.get(property_key)
        if scalar is None:
            raise RuntimeError("SchNetPack did not return scalar_representation")
        graph = _segment_mean(scalar, batch["batch"], batch["lattice"].shape[0])
        return self.head(graph)


class OfficialMatGLM3GNet(nn.Module):
    """Official MatGL M3GNet model, adapted to the fixed NFE periodic graph batch."""

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
            "ntargets": 4,
        }
        signature = inspect.signature(M3GNet.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        self.model = M3GNet(**kwargs)
        self.element_to_index = {symbol: i for i, symbol in enumerate(element_types)}

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise RuntimeError("MatGL 4.x requires torch-geometric") from exc
        from pymatgen.core import Element

        z_symbols = [Element.from_Z(int(z)).symbol for z in batch["z"].detach().cpu().tolist()]
        node_type = torch.tensor(
            [self.element_to_index[s] for s in z_symbols],
            dtype=torch.long,
            device=batch["z"].device,
        )
        atom_lattice = batch["lattice"][batch["batch"]]
        pos = torch.einsum("ni,nij->nj", batch["frac_pos"], atom_lattice)
        source, destination = batch["edge_index"]
        edge_lattice = batch["lattice"][batch["batch"][destination]]
        offshift = torch.einsum("ei,eij->ej", batch["edge_shift"], edge_lattice)
        data = Data(
            node_type=node_type,
            z=node_type,
            pos=pos,
            edge_index=torch.stack([destination, source], dim=0),
            pbc_offshift=offshift,
            batch=batch["batch"],
            num_graphs=int(batch["lattice"].shape[0]),
        )
        out = self.model(data)
        if isinstance(out, dict):
            out = out.get("output", out.get("pred", out.get("energy")))
        out = torch.as_tensor(out, device=batch["z"].device)
        if out.ndim == 1:
            out = out.unsqueeze(0) if batch["lattice"].shape[0] == 1 else out.view(-1, 4)
        if out.shape[-1] != 4:
            raise RuntimeError(f"MatGL M3GNet expected 4 outputs, got {tuple(out.shape)}")
        return {"class_logits": out[:, :3], "score": out[:, 3]}


class OfficialALIGNN(nn.Module):
    """Official ALIGNN line-graph model with common NFE four-output regression head."""

    def __init__(self, hidden_dim: int, num_layers: int, cutoff: float, max_neighbors: int) -> None:
        super().__init__()
        try:
            import dgl
            from alignn.graphs import Graph
            from alignn.models.alignn import ALIGNN, ALIGNNConfig
        except ImportError as exc:
            raise RuntimeError(
                "ALIGNN backend requires an isolated environment with alignn==2026.5.20"
            ) from exc
        self.dgl = dgl
        self.Graph = Graph
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=int(num_layers),
            gcn_layers=int(num_layers),
            hidden_features=int(hidden_dim),
            output_features=4,
            classification=False,
        )
        self.model = ALIGNN(config)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)

    def _graph(self, file_path: str, device: torch.device):
        from jarvis.core.atoms import Atoms

        structure = Structure.from_file(file_path)
        atoms = Atoms(
            lattice_mat=structure.lattice.matrix,
            coords=structure.frac_coords,
            elements=[site.specie.symbol for site in structure],
            cartesian=False,
        )
        g, lg = self.Graph.atom_dgl_multigraph(
            atoms=atoms,
            neighbor_strategy="k-nearest",
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            atom_features="cgcnn",
            compute_line_graph=True,
            use_canonize=True,
        )
        return g.to(device), lg.to(device), torch.tensor(
            structure.lattice.matrix, dtype=torch.float32, device=device
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        device = batch["z"].device
        triples = [self._graph(path, device) for path in batch["file_paths"]]
        g = self.dgl.batch([x[0] for x in triples])
        lg = self.dgl.batch([x[1] for x in triples])
        lattice = torch.stack([x[2] for x in triples])
        out = self.model((g, lg, lattice))
        if out.ndim == 1:
            out = out.unsqueeze(0)
        return {"class_logits": out[:, :3], "score": out[:, 3]}


class OfficialCGCNN(nn.Module):
    """Original txie-93 CGCNN network with a project data adapter and four-output head."""

    def __init__(
        self,
        cgcnn_repo: str | Path,
        atom_init_json: str | Path,
        hidden_dim: int,
        num_layers: int,
        cutoff: float,
        max_neighbors: int,
    ) -> None:
        super().__init__()
        repo = Path(cgcnn_repo).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance
            from cgcnn.model import CrystalGraphConvNet
        except ImportError as exc:
            raise RuntimeError(
                "CGCNN backend requires a checkout of https://github.com/txie-93/cgcnn"
            ) from exc
        self.atom_init = AtomCustomJSONInitializer(str(Path(atom_init_json).resolve()))
        self.gdf = GaussianDistance(dmin=0, dmax=float(cutoff), step=0.2)
        atom_fea_len = len(self.atom_init.get_atom_fea(1))
        nbr_fea_len = len(self.gdf.expand([0.0])[0])
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=int(hidden_dim),
            n_conv=int(num_layers),
            h_fea_len=int(hidden_dim),
            n_h=1,
            classification=False,
        )
        self.model.fc_out = nn.Linear(self.model.fc_out.in_features, 4)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)

    def _crystal(self, file_path: str, device: torch.device):
        structure = Structure.from_file(file_path)
        atom_fea = torch.tensor(
            [self.atom_init.get_atom_fea(int(site.specie.Z)) for site in structure],
            dtype=torch.float32,
            device=device,
        )
        all_nbrs = structure.get_all_neighbors(self.cutoff, include_index=True)
        nbr_idx = []
        nbr_dist = []
        for nbrs in all_nbrs:
            ordered = sorted(nbrs, key=lambda x: float(x.nn_distance))[: self.max_neighbors]
            indices = [int(x.index) for x in ordered]
            distances = [float(x.nn_distance) for x in ordered]
            while len(indices) < self.max_neighbors:
                indices.append(0)
                distances.append(self.cutoff + 1.0)
            nbr_idx.append(indices)
            nbr_dist.append(distances)
        nbr_fea_idx = torch.tensor(nbr_idx, dtype=torch.long, device=device)
        import numpy as np

        expanded = self.gdf.expand(np.asarray(nbr_dist, dtype=float))
        nbr_fea = torch.tensor(expanded, dtype=torch.float32, device=device)
        return atom_fea, nbr_fea, nbr_fea_idx

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        device = batch["z"].device
        crystals = [self._crystal(path, device) for path in batch["file_paths"]]
        atom_parts, nbr_parts, idx_parts, crystal_atom_idx = [], [], [], []
        offset = 0
        for atom_fea, nbr_fea, nbr_idx in crystals:
            n = atom_fea.shape[0]
            atom_parts.append(atom_fea)
            nbr_parts.append(nbr_fea)
            idx_parts.append(nbr_idx + offset)
            crystal_atom_idx.append(torch.arange(offset, offset + n, device=device))
            offset += n
        out = self.model(
            torch.cat(atom_parts, dim=0),
            torch.cat(nbr_parts, dim=0),
            torch.cat(idx_parts, dim=0),
            crystal_atom_idx,
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
) -> nn.Module:
    if name == "schnet_official":
        return OfficialSchNetPack(hidden_dim, num_layers, cutoff)
    if name == "m3gnet_official":
        return OfficialMatGLM3GNet(element_types, hidden_dim, num_layers, cutoff)
    if name == "alignn_official":
        return OfficialALIGNN(hidden_dim, num_layers, cutoff, max_neighbors)
    if name == "cgcnn_official":
        if not cgcnn_repo or not cgcnn_atom_init:
            raise ValueError("cgcnn_official requires --cgcnn-repo and --cgcnn-atom-init")
        return OfficialCGCNN(
            cgcnn_repo,
            cgcnn_atom_init,
            hidden_dim,
            num_layers,
            cutoff,
            max_neighbors,
        )
    raise ValueError(f"unknown official backend: {name}")
