"""Heatmap-regression CNN for face alignment.

Requires PyTorch (`pip install torch`). This is the deep counterpart to the
NumPy logic in heatmap.py. It is NOT executed in the reference run here because
the assignment image data is not bundled; supply the data via dataset.py and
call run_task2.py.

Architecture (a compact encoder-decoder / "hourglass-lite", W11_L21 lineage):

    input 1x64x64
      -> [Conv3-32, BN, ReLU] x2, MaxPool          -> 32x32x32
      -> [Conv3-64, BN, ReLU] x2, MaxPool          -> 64x16x16   (bottleneck)
      -> Upsample, [Conv3-32, BN, ReLU]            -> 32x32x32
      -> Upsample, [Conv3-16, BN, ReLU]            -> 16x64x64
      -> Conv1 -> K heatmaps (Kx64x64)

Output heatmaps are decoded with soft-argmax (differentiable) so the loss can
combine a pixel-wise heatmap MSE (against Gaussian targets) with a coordinate
loss on the decoded points -- a standard and robust recipe. The lecture
sanctions MSE / squared-Euclidean as the loss; we keep MSE primary and note
Huber/L1 as the lecture-acknowledged "more robust" alternative.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if _HAS_TORCH:

    def conv_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class HeatmapNet(nn.Module):
        def __init__(self, n_landmarks: int = 5, in_ch: int = 1, base: int = 32):
            super().__init__()
            self.enc1 = conv_block(in_ch, base)
            self.enc2 = conv_block(base, base * 2)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.dec1 = conv_block(base * 2, base)
            self.dec2 = conv_block(base, base // 2)
            self.head = nn.Conv2d(base // 2, n_landmarks, 1)

        def forward(self, x):                       # x: (B,1,64,64)
            x = self.enc1(x)                        # 32 x 64 x 64
            x = self.enc2(self.pool(x))             # 64 x 32 x 32
            x = self.pool(x)                        # 64 x 16 x 16  (bottleneck)
            x = self.dec1(self.up(x))               # 32 x 32 x 32
            x = self.dec2(self.up(x))               # 16 x 64 x 64
            return self.head(x)                     # (B,K,64,64)

    def gaussian_heatmaps(pts, hw, sigma: float = 1.5):
        """Batched Gaussian targets: (B,K,2) coords -> (B,K,H,W) maps peaking at 1.

        The torch twin of ``heatmap.make_heatmaps``, built on-device so targets
        cost nothing per epoch instead of materialising ~1 GB of float32 up
        front. ``tests_sanity.py`` asserts the two agree.
        """
        H, W = hw
        ys = torch.arange(H, device=pts.device, dtype=pts.dtype).view(1, 1, H, 1)
        xs = torch.arange(W, device=pts.device, dtype=pts.dtype).view(1, 1, 1, W)
        px = pts[..., 0].unsqueeze(-1).unsqueeze(-1)
        py = pts[..., 1].unsqueeze(-1).unsqueeze(-1)
        return torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))

    def soft_argmax2d(heatmaps, beta: float = 10.0):
        """Differentiable coordinate decode. Returns (B,K,2) in heatmap pixels."""
        B, K, H, W = heatmaps.shape
        p = F.softmax(heatmaps.view(B, K, -1) * beta, dim=-1).view(B, K, H, W)
        xs = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype)
        ys = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype)
        ex = (p.sum(dim=2) * xs).sum(dim=-1)
        ey = (p.sum(dim=3) * ys).sum(dim=-1)
        return torch.stack([ex, ey], dim=-1)

    def combined_loss(pred_hm, target_hm, pred_xy, target_xy, w_coord: float = 0.1):
        """Heatmap MSE + weighted coordinate MSE on the soft-argmax output.

        The coordinate term is measured in *grid-relative* units (coords divided
        by the heatmap width/height) rather than pixels. In pixels it would start
        at ~10^3 against a heatmap MSE of ~10^-2, so no sane ``w_coord`` could
        keep the heatmap term relevant; normalised, both terms are O(10^-2) at
        initialisation and ``w_coord`` does what it says.
        """
        H, W = pred_hm.shape[-2:]
        s = pred_hm.new_tensor([float(W), float(H)])
        return (F.mse_loss(pred_hm, target_hm)
                + w_coord * F.mse_loss(pred_xy / s, target_xy / s))

else:  # pragma: no cover - informative stub when torch is absent
    HeatmapNet = None

    def gaussian_heatmaps(*a, **k):
        raise ImportError("PyTorch not installed. `pip install torch` to use the CNN.")

    def soft_argmax2d(*a, **k):
        raise ImportError("PyTorch not installed. `pip install torch` to use the CNN.")

    def combined_loss(*a, **k):
        raise ImportError("PyTorch not installed. `pip install torch` to use the CNN.")
