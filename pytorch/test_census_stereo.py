"""Smoke tests for census_stereo.py. Run: python test_census_stereo.py  (or pytest)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import census_stereo as cs  # noqa: E402

CPU = torch.device('cpu')
CENSUS, AGG = 5, 5


def make_pair(d, h=40, w=100, seed=0):
    """right(x) = left(x + d): content moves LEFT by d in the right image."""
    g = torch.Generator().manual_seed(seed)
    left = torch.rand(1, 1, h, w + 0, generator=g) * 255
    right = torch.rand(1, 1, h, w, generator=g) * 255
    right[..., :w - d] = left[..., d:]
    return left, right


def margin():
    return CENSUS // 2 + AGG // 2


def test_centre_bit_never_set():
    img = torch.rand(2, 1, 20, 30) * 255
    for k in (3, 5, 11):
        bits = cs.census_transform(img, k)
        assert bits.shape == (2, 20, 30, k * k)
        assert not bits[..., (k * k) // 2].any()
        assert bits.any()


def _check(d, min_disp, max_disp, sub_pixel, lr):
    left, right = make_pair(d)
    if lr:
        disp_L, disp_R = cs.get_disparity(left, right, min_disp, max_disp, AGG, CENSUS, sub_pixel, CPU, 2)
    else:
        disp_L, disp_R = cs.stereo_matching(left, right, min_disp, max_disp, AGG, CENSUS, sub_pixel, CPU)
    m, w = margin(), left.size(3)
    # disp_R valid where right(x) <-> left(x+d) is inside the image; disp_L where x-d >= 0
    r = disp_R[0, m:-m, m:w - d - m]
    l = disp_L[0, m:-m, d + m:w - m]
    tol = 0.5 if sub_pixel else 1e-6
    assert (r - d).abs().max() <= tol, (d, (r - d).abs().max())
    assert (l - d).abs().max() <= tol, (d, (l - d).abs().max())
    print('d=%d min=%d sub=%s lr=%s  maxerr R=%.3f L=%.3f' % (
        d, min_disp, sub_pixel, lr, (r - d).abs().max(), (l - d).abs().max()))


def test_constant_disparity():
    _check(7, 0, 15, False, False)


def test_zero_disparity():
    _check(0, 0, 10, False, False)


def test_min_disp_offset():
    _check(5, 2, 12, False, False)


def test_subpixel_on():
    _check(6, 0, 12, True, False)


def test_full_pipeline_with_lr_check():
    _check(6, 0, 12, False, True)
    _check(6, 0, 12, True, True)


def test_batch_matches_single():
    l0, r0 = make_pair(4, seed=1)
    l1, r1 = make_pair(9, seed=2)
    bl, br = torch.cat([l0, l1]), torch.cat([r0, r1])
    bL, bR = cs.get_disparity(bl, br, 0, 12, AGG, CENSUS, True, CPU, 2)
    for i, (l, r) in enumerate(((l0, r0), (l1, r1))):
        sL, sR = cs.get_disparity(l, r, 0, 12, AGG, CENSUS, True, CPU, 2)
        assert torch.allclose(bL[i], sL[0]) and torch.allclose(bR[i], sR[0])


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('PASS', t.__name__)
    print('all %d tests passed' % len(tests))
