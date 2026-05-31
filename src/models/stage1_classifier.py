"""Stage 1 action-type classifier.

3-layer MLP per the roadmap §Phase 3 spec. Operates on cached, pooled features
from a frozen Qwen2-VL backbone — the VLM is not fine-tuned at this stage.

Two feature modes supported:
  * "text_only"   — features were pooled from a forward pass with no image.
  * "vision_text" — features were pooled from a forward pass with image+text.

The model itself does not know the mode; that distinction lives at the feature
extraction step. This file is a plain classifier head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Stage1Config:
    feature_dim: int = 1536  # Qwen2-VL-2B hidden_size; 3584 for the 7B stretch run
    hidden_dim: int = 1024
    num_classes: int = 8
    dropout: float = 0.1


class Stage1Classifier(nn.Module):
    """3-layer MLP on pooled VLM features.

    Architecture matches the roadmap §Phase 3 sketch:
        Linear(feature_dim -> hidden_dim)
        GELU
        Dropout(p)
        Linear(hidden_dim -> hidden_dim // 4)
        GELU
        Linear(hidden_dim // 4 -> num_classes)
    """

    def __init__(self, cfg: Stage1Config | None = None, **overrides) -> None:
        super().__init__()
        if cfg is None:
            cfg = Stage1Config()
        if overrides:
            cfg = Stage1Config(**{**cfg.__dict__, **overrides})
        self.cfg = cfg
        bottleneck = max(cfg.num_classes, cfg.hidden_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, cfg.num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
