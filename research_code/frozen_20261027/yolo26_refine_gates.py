"""Contained identity adapters inspired by Strip R-CNN, DEGConv and PConv.

These are experimental adaptations, NOT canonical reproductions of those papers
or redistributed copies of the commercial module pack. All leave the pretrained
backbone and Segment26 tensor interface intact. PConv reduces adapter cost only;
it does not compress the original detector. No global package monkeypatching.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Segment26


class IdentityFeatureGate(nn.Module):
    """Zero-initialized bounded spatial/channel gate, initially exact identity."""

    def __init__(self, channels: int, kind: str, kernel: int = 7):
        super().__init__()
        self.channels = channels
        self.kind = kind
        self.kernel = kernel
        if kernel % 2 != 1 or kernel < 3:
            raise ValueError("Expected odd spatial kernel >=3")
        if kind == "strip":
            self.features = nn.Sequential(
                nn.Conv2d(channels, channels, (1, kernel), padding=(0, kernel // 2), groups=channels),
                nn.Conv2d(channels, channels, (kernel, 1), padding=(kernel // 2, 0), groups=channels),
                nn.SiLU(),
            )
            self.logits = nn.Conv2d(channels, channels, 1)
        elif kind == "partial":
            self.partial_channels = max(channels // 4, 1)
            self.partial_conv = nn.Conv2d(self.partial_channels, self.partial_channels, 3, padding=1)
            self.logits = nn.Conv2d(channels, channels, 1)
        elif kind == "deg":
            # Vectorized orientation histograms replace per-cell Python bincount.
            # This simplified gate uses global directional context, not the full
            # paper's encoder or the commercial pack's exact DEGConv wrapper.
            self.bins = 18
            self.hog_projection = nn.Conv2d(self.bins, channels, 1)
            self.horizontal = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels)
            self.vertical = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels)
            self.logits = nn.Conv2d(channels, channels, 1)
            self.register_buffer("sobel_x", torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).reshape(1, 1, 3, 3))
            self.register_buffer("sobel_y", torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).reshape(1, 1, 3, 3))
        else:
            raise ValueError(kind)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def orientation_histogram(self, x):
        # The hard-bin directional prior intentionally carries no input gradient;
        # learned projection / strip path / gating remain trainable.
        with torch.no_grad():
            image = x.detach().float().mean(1, keepdim=True)
            dx = F.conv2d(image, self.sobel_x.float(), padding=1)
            dy = F.conv2d(image, self.sobel_y.float(), padding=1)
            angle = torch.atan2(dy, dx).abs()
            index = (angle * (self.bins / torch.pi)).long().clamp(0, self.bins - 1).flatten(1)
            hist = image.new_zeros((x.shape[0], self.bins))
            hist.scatter_add_(1, index, torch.ones_like(index, dtype=hist.dtype))
            hist = hist / index.shape[1]
        return hist.to(x.dtype)[:, :, None, None]

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("Adapter channel/shape mismatch")
        if self.kind == "strip":
            feature = self.features(x)
        elif self.kind == "partial":
            left, right = x.split([self.partial_channels, self.channels - self.partial_channels], dim=1)
            feature = F.silu(torch.cat((self.partial_conv(left), right), dim=1))
        else:
            context = self.hog_projection(self.orientation_histogram(x))
            feature = F.silu(self.horizontal(x) + self.vertical(x) + context)
        # 0.8..1.2 modulation; zero logits are exactly identity, not sigmoid=.5.
        return x * (1.0 + 0.2 * torch.tanh(self.logits(feature)))


class RefineGateSegment26(Segment26):
    def forward(self, x):
        return super().forward([gate(value) for gate, value in zip(self.refine_gates, x)])


def install(model, kind: str, kernel: int = 7):
    head = model.model[-1]
    if not isinstance(head, Segment26):
        raise ValueError("Requires Segment26")
    if kind == "control":
        if isinstance(head, RefineGateSegment26):
            raise ValueError("Control cannot contain adapters")
        return model
    if isinstance(head, RefineGateSegment26):
        raise ValueError("Refuse double adapter installation")
    # Avoid changing the paired training stream just by initializing a module.
    state = torch.get_rng_state()
    channels = [layer[0].conv.in_channels for layer in head.cv2]
    try:
        gates = nn.ModuleList([IdentityFeatureGate(c, kind, kernel) for c in channels])
    finally:
        torch.set_rng_state(state)
    head.__class__ = RefineGateSegment26
    head.refine_gates = gates
    head.refine_kind = kind
    return model
