"""Heatmap regression CNN for face alignment.

Install PyTorch before you use this module. Use the command
`pip install torch`.

This module is the deep equivalent of the NumPy code in heatmap.py.

The reference run does not execute this module, because the image data of the
assignment is not in the archive. Give the data to dataset.py, then run
run_task2.py.

The architecture is a compact encoder-decoder. It is a small hourglass network
of the type in W11_L21:

    input 1x64x64
      -> [Conv3-32, BN, ReLU] x2, MaxPool          -> 32x32x32
      -> [Conv3-64, BN, ReLU] x2, MaxPool          -> 64x16x16   (bottleneck)
      -> Upsample, [Conv3-32, BN, ReLU]            -> 32x32x32
      -> Upsample, [Conv3-16, BN, ReLU]            -> 16x64x64
      -> Conv1 -> K heatmaps (Kx64x64)

The soft-argmax decodes the output heatmaps. This decode is differentiable.
Thus the loss can contain two terms. The first term is the pixel-wise MSE
between the heatmaps and the Gaussian targets. The second term is a coordinate
loss on the decoded landmarks. This combination is usual and robust.

The lecture permits the MSE or the squared Euclidean distance as the loss.
This module uses the MSE. The lecture gives the Huber loss and the L1 loss as
more robust alternatives.
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
        """Make Gaussian targets for a batch.

        The function changes (B,K,2) coordinates into (B,K,H,W) heatmaps. Each
        heatmap has a peak value of 1.

        This function is the PyTorch equivalent of heatmap.make_heatmaps. It
        makes the targets on the device. Thus each epoch does not use memory
        for the targets. The alternative is a store of approximately 1 GB of
        float32 values. The file tests_sanity.py makes sure that the two
        functions agree.
        """
        H, W = hw
        ys = torch.arange(H, device=pts.device, dtype=pts.dtype).view(1, 1, H, 1)
        xs = torch.arange(W, device=pts.device, dtype=pts.dtype).view(1, 1, 1, W)
        px = pts[..., 0].unsqueeze(-1).unsqueeze(-1)
        py = pts[..., 1].unsqueeze(-1).unsqueeze(-1)
        return torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))

    def soft_argmax2d(heatmaps, beta: float = 10.0):
        """Decode the heatmaps to coordinates.

        This decode is differentiable. The function returns a (B,K,2) tensor in
        heatmap pixels.
        """
        B, K, H, W = heatmaps.shape
        p = F.softmax(heatmaps.view(B, K, -1) * beta, dim=-1).view(B, K, H, W)
        xs = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype)
        ys = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype)
        ex = (p.sum(dim=2) * xs).sum(dim=-1)
        ey = (p.sum(dim=3) * ys).sum(dim=-1)
        return torch.stack([ex, ey], dim=-1)

    def combined_loss(pred_hm, target_hm, pred_xy, target_xy, w_coord: float = 0.1):
        """Add the heatmap MSE to the weighted coordinate MSE.

        The coordinate MSE uses the output of the soft-argmax.

        The function measures the coordinate term in grid-relative units. It
        divides each coordinate by the width or the height of the heatmap. The
        two possible scales are different:

          * In pixels the coordinate term starts at approximately 10^3. The
            heatmap term starts at approximately 10^-2. No value of w_coord
            keeps the heatmap term important.
          * In grid-relative units both terms start at approximately 10^-2.
            Then w_coord has the correct effect.
        """
        H, W = pred_hm.shape[-2:]
        s = pred_hm.new_tensor([float(W), float(H)])
        return (F.mse_loss(pred_hm, target_hm)
                + w_coord * F.mse_loss(pred_xy / s, target_xy / s))

else:  # pragma: no cover - these stubs give an error message if torch is absent
    HeatmapNet = None

    def gaussian_heatmaps(*a, **k):
        raise ImportError("PyTorch not installed. `pip install torch` to use the CNN.")

    def soft_argmax2d(*a, **k):
        raise ImportError("PyTorch not installed. `pip install torch` to use the CNN.")

    def combined_loss(*a, **k):
        raise ImportError("PyTorch not installed. `pip install torch` to use the CNN.")
