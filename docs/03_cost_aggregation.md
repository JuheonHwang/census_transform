# 3장. 비용 집계: box filter

## 목표

- 픽셀 단위 비용이 불안정한 이유와, 주변 픽셀의 비용을 합산(cost aggregation)해 안정화하는 원리를 이해합니다.
- box filter를 `conv2d`(합계, 평균이 아님)로 구현하는 방법과 경계 처리를 파악합니다.
- 집계 창 크기의 trade-off를 이해합니다.

## 개념 설명

2장에서 보았듯이 한 픽셀의 census 비트열(7x7이면 49비트)만으로는 올바른 disparity에서도 비용이 충분히 낮지 않거나 다른 disparity와 구분이 안 될 수 있습니다. 가정을 하나 추가합니다. **이웃 픽셀은 대체로 같은 disparity를 가진다**(국소적 smoothness, fronto-parallel 가정). 그러면 각 disparity $d$마다 픽셀 주변 $w \times w$ 영역의 비용을 더해서 비교하는 것이 더 신뢰할 만합니다.

이 단순한 방법이 **box filter 집계**입니다. 각 disparity 슬라이스(`(H, W)` 비용 맵)에 독립적으로 $w \times w$ 합계 필터를 적용합니다.

창 크기 trade-off는 다음과 같습니다.

- 작게: 디테일과 물체 경계가 유지되지만 노이즈에 취약합니다.
- 크게: 매끄럽고 안정적이지만 경계가 뭉개지고(foreground fattening) 얇은 구조가 사라집니다. 또 창 안에 서로 다른 depth가 섞이면 틀린 disparity를 고릅니다.

## 수식

$$
C(y, x, d) = \sum_{i=-a}^{a} \sum_{j=-a}^{a} c(y+i,\, x+j,\, d), \qquad a = \lfloor w/2 \rfloor
$$

**평균이 아니라 합계**입니다. 모든 disparity에 같은 상수가 곱해지는 셈이므로 argmin 결과는 달라지지 않지만, 비용의 절대 크기(예: 11x11 창, 7x7 census의 최대 $121 \times 49 = 5929$)는 달라집니다. 이후 sub-pixel 보간(4장)은 비용의 *상대* 값만 쓰므로 영향이 없습니다.

### 경계

영상 밖은 비용을 0으로 간주하므로 경계 근처의 합은 더 적은 항으로 이루어집니다. $11 \times 11$ 창이면 모서리 픽셀은 $6 \times 6 = 36$개 항, 경계에서 2픽셀 떨어진 곳은 $8 \times 8 = 64$개 항, 안쪽은 121개 항입니다(아래 실행 결과 참고). 경계 픽셀은 비용 크기가 작아도 비교는 같은 픽셀 안의 $d$ 사이에서만 이루어지므로 argmin에는 문제가 없습니다.

## 코드

### PyTorch

```python
    box = torch.ones((1, 1, agg_size, agg_size), device=census_ref.device)
    volume = torch.empty((bz, h, w, n), dtype=torch.float32, device=census_ref.device)
    for i in range(n):
        tgt = shift_census(census_tgt, min_disp + i, direction)
        hamming = torch.bitwise_xor(census_ref, tgt).sum(dim=3, dtype=torch.float32)  # (B,H,W)
        volume[..., i] = F.conv2d(hamming.unsqueeze(1), box, padding='same')[:, 0]
    return volume
```

[pytorch/census_stereo.py:87-93](../pytorch/census_stereo.py#L87-L93)

`box`는 전부 1인 `agg_size x agg_size` 커널이고, `conv2d`의 `padding='same'`은 0 padding으로 출력 크기를 유지합니다. 커널이 모두 1이므로 이것이 곧 합계 필터입니다. 집계는 disparity 루프 안에서 슬라이스마다 한 번씩 적용되어 `volume[..., i]`에 저장됩니다. 결과 `volume`이 집계된 cost volume `(B, H, W, D)`입니다.

### MATLAB

```matlab
        sch(:,:,d+1)= imfilter(sum(z, 3), ones(ws));
```

[matlab/imstereo_sch_stereo.m:23](../matlab/imstereo_sch_stereo.m#L23)

`imfilter`의 기본 경계 옵션이 0 padding이므로 PyTorch와 동일합니다.

### CUDA

CUDA의 `VectorBox`는 픽셀당 스레드 하나가 창 안을 이중 루프로 직접 합산합니다. 창을 영상 경계에서 잘라내어(clip) 존재하는 픽셀만 더하므로 결과는 0 padding 합계와 같습니다. 이 커널은 픽셀당 $O(w^2)$ 연산인 단순한 구현이며, 7~9장의 CUDA 버전들(v1~v3)이 모두 같은 합산 방식을 씁니다. 버전 간 차이는 비용을 저장하고 최소를 찾는 방식에 있습니다.

```cpp
		const int height_begin = max(i - aggreate_half, 0);
		const int height_end = min(i + aggreate_half + 1, img_h);
		const int width_begin = max(j - aggreate_half, 0);
		const int width_end = min(j + aggreate_half + 1, img_w);

		for (int ii = height_begin; ii < height_end; ii++) {
			for (int jj = width_begin; jj < width_end; jj++) {
				cost_summation += Census_sum[ii * img_w + jj];
			}
		}
```

[cuda/src/v1_baseline.cu:144-153](../cuda/src/v1_baseline.cu#L144-L153)

## 직접 해보기

집계 전(`agg_size=1`)과 후(`agg_size=11`)의 비용 곡선과 WTA 결과를 비교합니다. CPU에서 수십 초 걸립니다.

```python
import sys; sys.path.insert(0, 'pytorch')
import torch, torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from census_stereo import census_transform, build_cost_volume, winner_take_all

L = read_image('data/sample/left.png', ImageReadMode.GRAY).unsqueeze(0)
R = read_image('data/sample/right.png', ImageReadMode.GRAY).unsqueeze(0)
cL, cR = census_transform(L, 7), census_transform(R, 7)

raw = build_cost_volume(cL, cR, 0, 48, 1, +1)     # agg_size=1: 집계 없음
agg = build_cost_volume(cL, cR, 0, 48, 11, +1)    # 11x11 box
print(agg.shape)

y, x = 360, 640
print('raw', raw[0, y, x].int().tolist()[:12], 'argmin', raw[0, y, x].argmin().item())
print('agg', agg[0, y, x].int().tolist()[:12], 'argmin', agg[0, y, x].argmin().item())

d_raw = winner_take_all(raw, 0, False)
d_agg = winner_take_all(agg, 0, False)
dx = lambda d: (d[..., 1:] - d[..., :-1]).abs().mean().item()
print('mean |d(x+1)-d(x)|  raw: %.2f  agg: %.2f' % (dx(d_raw), dx(d_agg)))

ones = torch.ones(1, 1, 720, 1280)
box = torch.ones(1, 1, 11, 11)
print(F.conv2d(ones, box, padding='same')[0, 0, [0, 2, 5, 360], [0, 2, 5, 640]].tolist())
```

실행 결과는 다음과 같습니다.

```
torch.Size([1, 720, 1280, 49])
raw [24, 26, 16, 15, 17, 11, 11, 14, 10, 7, 8, 17] argmin 9
agg [2354, 2259, 2084, 1949, 1768, 1466, 1633, 1763, 1752, 1766, 1864, 2002] argmin 5
mean |d(x+1)-d(x)|  raw: 4.85  agg: 0.77
[36.0, 64.0, 121.0, 121.0]
```

집계 후에는 이웃 픽셀 간 disparity 차이의 평균이 4.85에서 0.77로 크게 줄어듭니다(WTA 결과가 매끄러워짐). 집계 비용 곡선은 $d=5$ 부근에서 뚜렷한 골을 가지고(1466, 양옆 1768/1633), 이것이 4장의 sub-pixel 보간 입력이 됩니다. `agg_size`를 3, 7, 15 등으로 바꿔 직접 비교해 보세요.

## 구현별 차이

| 항목 | MATLAB | PyTorch | CUDA |
|---|---|---|---|
| 방법 | `imfilter(..., ones(ws))` | `F.conv2d` (ones 커널) | 직접 이중 루프 합산 (`VectorBox`) |
| 경계 | 0 padding | 0 padding (`padding='same'`) | 창을 경계에서 clip |
| 정규화 | 합계 | 합계 | 합계 |
| 경계 결과 | 동일 (0 padding = clip) | 동일 | 동일 |
| 짝수 크기 | 비대칭 | `padding='same'`이 비대칭 padding | $w-1$과 같게 동작 |

세 구현 모두 비용의 합이므로 결과가 같은 단위입니다. 다만 CUDA에서는 2장에서 언급한 popcount 부분 때문에 픽셀 비용 자체가 다를 수 있습니다. 집계 창 크기는 항상 홀수로 쓰는 것이 안전합니다. CUDA 버전별 비용 저장 방식은 [7장](07_cuda_baseline.md), [8장](08_cuda_kernel_fusion.md), [9장](09_cuda_rolling_buffer.md)을 참고하세요.

---

이전: [2장. 매칭 비용](02_matching_cost.md) | 다음: [4장. Disparity 선택: WTA와 sub-pixel 보간](04_disparity_subpixel.md)
