# ==============================================================================
# 中文概述：为最终 surface/manifold 表面条件流提供连续时间 Fourier 嵌入。
# English overview: Provide continuous-time Fourier embeddings for the final
# surface/manifold surface-aware conditional flow.
#
# 中文输入：形状为 `[batch]` 的归一化 ODE 时间。
# English inputs: Normalized ODE time with shape `[batch]`.
# 中文输出：正弦/余弦时间特征。
# English outputs: Concatenated sine/cosine time features.
#
# Author: Ruck
# Generated: 2026-07-30 10:25:00 Asia/Shanghai
# ==============================================================================

from __future__ import annotations

import math

import torch
import torch.nn as nn


# 中文：固定对数频率使网络同时感知粗粒度与细粒度流时间。
# English: Log-spaced fixed frequencies expose both coarse and fine flow time scales.
class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * math.pi * time.unsqueeze(-1) * self.frequencies
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
