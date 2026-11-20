"""Census-transform stereo matching in PyTorch (port of the MATLAB reference).

Pipeline (one function per stage):
    1. census_transform        image            -> per-pixel bit strings
    2. build_cost_volume       two census maps  -> XOR + popcount cost, box-aggregated
    3. (aggregation is part of build_cost_volume: box filter via conv2d)
    4. winner_take_all         cost volume      -> disparity (+ parabolic sub-pixel)
    5. left_right_check        disp_L, disp_R   -> inconsistent pixels set to 0
`stereo_matching` runs stages 1-4 for both views, `get_disparity` adds stage 5.

Disparity convention (rectified pair, positive disparities):
    disp_L[y, x] = d  means  left(y, x)  matches  right(y, x - d)
    disp_R[y, x] = d  means  right(y, x) matches  left(y, x + d)
Both maps are positive for a normal pair (objects shift LEFT in the right image).
Invalid pixels after the left-right check are 0.

Inputs are (B, 1, H, W) grayscale tensors (any dtype; compared as float).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Stage 1: census transform
# --------------------------------------------------------------------------
def census_transform(img, census_size):
    """Census transform.

    img: (B, 1, H, W) grayscale. census_size: odd window size k.
    Returns a bool tensor (B, H, W, k*k); bit j is True if window pixel j is
    strictly brighter than the window centre. The centre bit (index
    (k*k)//2) is therefore always False. Borders are zero padded.
    """
    if census_size % 2 == 0:
        raise ValueError("census_size must be odd")
    if img.dim() != 4 or img.size(1) != 1:
        raise ValueError("img must have shape (B, 1, H, W)")
    bz, _, h, w = img.shape
    r = census_size // 2
    padded = F.pad(img.float(), (r, r, r, r))
    blocks = F.unfold(padded, kernel_size=census_size)          # (B, k*k, H*W)
    centre = (census_size ** 2) // 2                            # 0-based centre
    bits = blocks > blocks[:, centre:centre + 1, :]             # (B, k*k, H*W)
    return bits.reshape(bz, census_size ** 2, h, w).permute(0, 2, 3, 1)


# --------------------------------------------------------------------------
# Stages 2-3: matching cost (XOR + popcount) and box-filter aggregation
# --------------------------------------------------------------------------
def shift_census(census, shift, direction):
    """Shift a census map along W by `shift` columns, zero-filling (no wrap).

    census: (B, H, W, K). direction=-1: out[..., x, :] = census[..., x+shift, :]
    (used for the right view); direction=+1: out[..., x, :] = census[..., x-shift, :]
    (used for the left view). Columns without a source are 0.
    """
    out = torch.zeros_like(census)
    w = census.size(2)
    if shift >= w:
        return out
    if shift == 0:
        return census.clone()
    if direction < 0:
        out[:, :, :w - shift, :] = census[:, :, shift:, :]
    else:
        out[:, :, shift:, :] = census[:, :, :w - shift, :]
    return out


def build_cost_volume(census_ref, census_tgt, min_disp, max_disp, agg_size, direction):
    """Aggregated census matching cost volume.

    census_ref, census_tgt: (B, H, W, K) bool. direction=-1 builds the volume
    for the right view as reference (target = left image), +1 for the left view
    as reference (target = right image).
    cost[b, y, x, i] = sum over an agg_size x agg_size box of
    popcount(ref_bits XOR shifted_tgt_bits) at disparity min_disp + i.
    Returns float32 (B, H, W, max_disp - min_disp + 1).
    """
    bz, h, w, _ = census_ref.shape
    n = max_disp - min_disp + 1
    box = torch.ones((1, 1, agg_size, agg_size), device=census_ref.device)
    volume = torch.empty((bz, h, w, n), dtype=torch.float32, device=census_ref.device)
    for i in range(n):
        tgt = shift_census(census_tgt, min_disp + i, direction)
        hamming = torch.bitwise_xor(census_ref, tgt).sum(dim=3, dtype=torch.float32)  # (B,H,W)
        volume[..., i] = F.conv2d(hamming.unsqueeze(1), box, padding='same')[:, 0]
    return volume


# --------------------------------------------------------------------------
# Stage 4: winner-take-all + parabolic sub-pixel refinement
# --------------------------------------------------------------------------
def refine_subpixel(idx, cost):
    """Parabolic sub-pixel refinement of integer WTA indices.

    idx: (B, H, W) long, argmin over the last axis of cost (B, H, W, N).
    Returns float32 (B, H, W): idx + (c[i+1]-c[i-1]) / (2*(2c[i]-c[i-1]-c[i+1]))
    for interior indices 0 < i < N-1 with a non-flat parabola; other pixels
    keep the integer index.
    """
    n = cost.size(3)
    out = idx.float()
    if n < 3:
        return out
    i = idx.clamp(1, n - 2).unsqueeze(-1)
    c_d = cost.gather(3, i)[..., 0]
    c_m = cost.gather(3, i - 1)[..., 0]
    c_p = cost.gather(3, i + 1)[..., 0]
    denom = 2 * (2 * c_d - c_m - c_p)
    valid = (idx > 0) & (idx < n - 1) & (denom != 0) & torch.isfinite(denom)
    offset = torch.where(valid, (c_p - c_m) / torch.where(valid, denom, torch.ones_like(denom)),
                         torch.zeros_like(denom))
    return out + offset


def winner_take_all(cost, min_disp, sub_pixel):
    """Pick the minimum-cost disparity per pixel.

    cost: (B, H, W, N). Returns float32 disparity (B, H, W) = argmin + min_disp,
    optionally refined to sub-pixel accuracy.
    """
    idx = torch.argmin(cost, dim=3)
    disp = refine_subpixel(idx, cost) if sub_pixel else idx.float()
    return disp + min_disp


def stereo_matching(left_img, right_img, min_disp, max_disp, aggreate_size, census_size, sub_pixel, device):
    """Stages 1-4 for both views.

    left_img, right_img: (B, 1, H, W). Returns (disp_L, disp_R), float32 (B, H, W),
    without the left-right check. See the module docstring for the convention.
    """
    left_img, right_img = left_img.to(device), right_img.to(device)
    census_L = census_transform(left_img, census_size)
    census_R = census_transform(right_img, census_size)

    cost_R = build_cost_volume(census_R, census_L, min_disp, max_disp, aggreate_size, -1)
    disp_R = winner_take_all(cost_R, min_disp, sub_pixel)
    del cost_R
    cost_L = build_cost_volume(census_L, census_R, min_disp, max_disp, aggreate_size, +1)
    disp_L = winner_take_all(cost_L, min_disp, sub_pixel)
    return disp_L, disp_R


# --------------------------------------------------------------------------
# Stage 5: left-right consistency check
# --------------------------------------------------------------------------
def _sample_rows_linear(disp, cols):
    """Linearly sample disp (B, H, W) at fractional columns cols (B, H, W)."""
    w = disp.size(2)
    cols = cols.clamp(0, w - 1)
    c0 = torch.floor(cols).long()
    c1 = (c0 + 1).clamp(max=w - 1)
    alpha = cols - c0
    return alpha * disp.gather(2, c1) + (1 - alpha) * disp.gather(2, c0)


def disparity_interp(disp_L, disp_R):
    """Warp each disparity map into the other view. Batch-correct.

    disp_L, disp_R: (B, H, W). Returns (disp_R2L, disp_L2R), both (B, H, W):
    disp_R2L[y, x] = disp_R sampled at x - disp_L[y, x] (linear, clamped);
    disp_L2R[y, x] = disp_L sampled at x + disp_R[y, x].
    """
    h, w = disp_L.shape[1:]
    x = torch.arange(w, device=disp_L.device, dtype=torch.float32).view(1, 1, w).expand_as(disp_L)
    return _sample_rows_linear(disp_R, x - disp_L), _sample_rows_linear(disp_L, x + disp_R)


def left_right_check(disp_L, disp_R, threshold):
    """Zero out pixels whose warped counterpart differs by more than threshold.

    disp_L, disp_R: (B, H, W). Returns new (disp_L, disp_R); inputs are unchanged.
    """
    disp_R2L, disp_L2R = disparity_interp(disp_L, disp_R)
    out_L = torch.where((disp_R2L - disp_L).abs() > threshold, torch.zeros_like(disp_L), disp_L)
    out_R = torch.where((disp_L2R - disp_R).abs() > threshold, torch.zeros_like(disp_R), disp_R)
    return out_L, out_R


def get_disparity(left_img, right_img, min_disp, max_disp, aggreate_size, census_size,
                  sub_pixel, device, threshold=10, visualize=False, out_dir='.'):
    """Full pipeline. Returns (disp_L, disp_R), float32 (B, H, W), invalid = 0.

    If visualize is True, saves normalised disp_L.png / disp_R.png into out_dir.
    """
    disp_L, disp_R = stereo_matching(left_img, right_img, min_disp, max_disp,
                                     aggreate_size, census_size, sub_pixel, device)
    disp_L, disp_R = left_right_check(disp_L, disp_R, threshold)
    if visualize:
        save_disparity(disp_L, disp_R, out_dir)
    return disp_L, disp_R


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def write_pfm(path, arr):
    """Write a 2-D float32 array as a little-endian PFM (bottom-to-top rows)."""
    arr = np.ascontiguousarray(np.flipud(arr.astype('<f4')))
    with open(path, 'wb') as f:
        f.write(b'Pf\n%d %d\n-1\n' % (arr.shape[1], arr.shape[0]))
        f.write(arr.tobytes())


def save_disparity(disp_L, disp_R, out_dir='.', save_npy=False, save_pfm=False):
    """Save disp_L.png / disp_R.png (divided by the map max) and optional float .npy/.pfm.

    For batch size > 1 files get a _<index> suffix.
    """
    from torchvision.utils import save_image
    os.makedirs(out_dir, exist_ok=True)
    for name, disp in (('L', disp_L), ('R', disp_R)):
        disp = disp.detach().cpu()
        for b in range(disp.size(0)):
            stem = os.path.join(out_dir, 'disp_%s%s' % (name, '' if disp.size(0) == 1 else '_%d' % b))
            d = disp[b]
            save_image((d / d.max().clamp(min=1e-6)).unsqueeze(0), stem + '.png')
            if save_npy:
                np.save(stem + '.npy', d.numpy())
            if save_pfm:
                write_pfm(stem + '.pfm', d.numpy())


def select_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main(argv=None):
    from torchvision.io import read_image, ImageReadMode
    p = argparse.ArgumentParser(description='Census-transform stereo matching (PyTorch).')
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--min-disp', type=int, default=0)
    p.add_argument('--max-disp', type=int, default=300)
    p.add_argument('--census-size', type=int, default=11)
    p.add_argument('--agg-size', type=int, default=11)
    p.add_argument('--lr-threshold', type=float, default=10)
    p.add_argument('--no-subpixel', action='store_true')
    p.add_argument('--device', default='auto', help="'auto', 'cpu', 'cuda', 'cuda:0', ...")
    p.add_argument('--out-dir', default='.')
    p.add_argument('--save-npy', action='store_true')
    p.add_argument('--save-pfm', action='store_true')
    a = p.parse_args(argv)

    device = select_device(a.device)
    left = read_image(a.left, ImageReadMode.GRAY).unsqueeze(0)
    right = read_image(a.right, ImageReadMode.GRAY).unsqueeze(0)

    start = time.time()
    disp_L, disp_R = get_disparity(left, right, a.min_disp, a.max_disp, a.agg_size, a.census_size,
                                   not a.no_subpixel, device, a.lr_threshold)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # CUDA ops are asynchronous; wait before reading the clock
    print('device=%s time=%.2fs' % (device, time.time() - start))
    save_disparity(disp_L, disp_R, a.out_dir, a.save_npy, a.save_pfm)


if __name__ == '__main__':
    main()
