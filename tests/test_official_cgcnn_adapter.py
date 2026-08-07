from __future__ import annotations

import torch
import torch.nn as nn

from training.baselines.official.backends import ragged_cgcnn_conv


class _ConvLike(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int) -> None:
        super().__init__()
        self.atom_fea_len = atom_dim
        self.nbr_fea_len = bond_dim
        self.fc_full = nn.Linear(2 * atom_dim + bond_dim, 2 * atom_dim)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_dim)
        self.bn2 = nn.BatchNorm1d(atom_dim)
        self.softplus2 = nn.Softplus()

    def dense_forward(
        self,
        atom_in_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
    ) -> torch.Tensor:
        n_atoms, n_neighbors = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(n_atoms, n_neighbors, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(n_atoms, n_neighbors, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_summed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_summed = self.bn2(nbr_summed)
        return self.softplus2(atom_in_fea + nbr_summed)


def test_ragged_cgcnn_matches_dense_upstream_operator_without_padding() -> None:
    torch.manual_seed(73)
    n_atoms, n_neighbors, atom_dim, bond_dim = 4, 3, 6, 5
    conv = _ConvLike(atom_dim, bond_dim).eval()
    atom = torch.randn(n_atoms, atom_dim)
    nbr_index = torch.tensor(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long
    )
    bond = torch.randn(n_atoms, n_neighbors, bond_dim)

    dense = conv.dense_forward(atom, bond, nbr_index)
    destination = torch.arange(n_atoms).repeat_interleave(n_neighbors)
    source = nbr_index.reshape(-1)
    ragged = ragged_cgcnn_conv(
        conv,
        atom,
        bond.reshape(-1, bond_dim),
        source,
        destination,
    )
    assert torch.allclose(dense, ragged, atol=1e-7, rtol=1e-7)
