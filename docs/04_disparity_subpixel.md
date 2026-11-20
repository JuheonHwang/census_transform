# 4장. Disparity 선택: WTA와 sub-pixel 보간

## 목표

- cost volume에서 픽셀마다 비용이 최소인 disparity를 고르는 **WTA**(winner-take-all)를 이해합니다.
- 정수 disparity의 양자화 오차를 줄이는 **parabola fitting sub-pixel 보간**의 유도와 수치 예시를 확인합니다.
- 경계 인덱스와 평평한 비용 곡선 같은 예외 처리를 구현별로 비교합니다.

## 개념 설명

3장까지 만든 cost volume `(H, W, D)`에서 각 픽셀의 비용 곡선 $C(d)$는 올바른 disparity 근처에서 골(valley)을 이룹니다. 가장 단순한 선택은 곡선의 최소 위치를 고르는 것입니다. 이를 **WTA**라 하며 이웃과의 협의 없이 각 픽셀이 독립적으로 결정합니다(smoothness는 3장의 집계로만 반영됩니다).

WTA의 결과는 정수입니다. 그러나 실제 disparity는 연속값이므로, 정수로 반올림하면 최대 0.5 px의 양자화 오차가 생기고 깊이 맵이 계단처럼 보입니다. **Sub-pixel 보간**은 최소점 $d^*$와 그 양옆 $d^*\!-\!1$, $d^*\!+\!1$의 세 비용에 포물선을 맞추고, 포물선의 꼭짓점을 실수 disparity로 사용합니다.

## 수식

세 점을 $x=-1, 0, 1$로 놓고 $c_- = C(d^*\!-\!1)$, $c_0 = C(d^*)$, $c_+ = C(d^*\!+\!1)$이라 하겠습니다. 포물선 $f(x) = ax^2 + bx + c_0$이 세 점을 지나려면

$$
a = \frac{c_- + c_+ - 2c_0}{2}, \qquad b = \frac{c_+ - c_-}{2}
$$

이고, 꼭짓점은 $x^* = -b/(2a)$입니다. 정리하면 다음과 같습니다.

$$
\hat d = d^* + \frac{c_+ - c_-}{2\,(2c_0 - c_- - c_+)}
$$

$d^*$가 최소이면 $c_0 \le c_\pm$이므로 보정량은 항상 $[-0.5,\ 0.5]$ 안에 있습니다. 분모가 0이면(세 비용이 일직선) 포물선이 정의되지 않습니다.

### 수치 예시

| $d$ | $d^*\!-\!1$ | $d^*$ | $d^*\!+\!1$ |
|:--:|:--:|:--:|:--:|
| 비용 | 30 | **20** | 26 |

$$
\text{offset} = \frac{26 - 30}{2\,(2\cdot 20 - 30 - 26)} = \frac{-4}{2\cdot(-16)} = 0.125
$$

검산: $a = (30+26-40)/2 = 8$, $b = (26-30)/2 = -2$, $x^* = 2/16 = 0.125$. 오른쪽 이웃(26)이 왼쪽 이웃(30)보다 낮으므로 꼭짓점이 오른쪽으로 조금 치우칩니다. 샘플 영상의 실제 값은 아래 "직접 해보기"에서 확인합니다.

```
cost
 30 |*
    |
 26 |           *
    |
 20 |      *            <- 정수 최솟값 d*
    +------+------+--
     d*-1   d*    d*+1      꼭짓점 = d* + 0.125
```

## 코드

### PyTorch

```python
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
```

[pytorch/census_stereo.py:99-119](../pytorch/census_stereo.py#L99-L119)

핵심은 `gather`로 픽셀마다 다른 인덱스 $i-1, i, i+1$의 비용을 한 번에 가져오는 것입니다. 인덱스가 범위를 벗어나지 않도록 `idx.clamp(1, n-2)`로 가져오되, 실제 보정은 `valid` 마스크(0 < idx < n-1, 분모 != 0)가 참인 픽셀에만 적용합니다. 그 외에는 `offset = 0`이라 정수 인덱스가 유지됩니다. `torch.where(valid, denom, ones)`는 0으로 나누기가 NaN을 만들지 않게 하는 안전장치입니다.

```python
def winner_take_all(cost, min_disp, sub_pixel):
    """Pick the minimum-cost disparity per pixel.

    cost: (B, H, W, N). Returns float32 disparity (B, H, W) = argmin + min_disp,
    optionally refined to sub-pixel accuracy.
    """
    idx = torch.argmin(cost, dim=3)
    disp = refine_subpixel(idx, cost) if sub_pixel else idx.float()
    return disp + min_disp
```

[pytorch/census_stereo.py:122-130](../pytorch/census_stereo.py#L122-L130)

`argmin`은 비용 축(마지막 축)에서 최솟값의 인덱스를 구합니다. 인덱스는 0부터 시작하므로 `min_disp`를 더해 실제 disparity로 바꿉니다. 동점일 때 어느 인덱스를 고르는지는 PyTorch가 보장하지 않습니다.

### MATLAB

```matlab
    [costR,dmapR]=min(sch,[],3);
    
    % SUBPIXEL
    if sub_pixel_refine
        cost = reshape(sch, [], diff(drange)+1);
        tar_idx = (dmapR(:)>1 & dmapR(:)<=diff(drange));
```

[matlab/imstereo_sch_stereo.m:25-30](../matlab/imstereo_sch_stereo.m#L25-L30)

```matlab
        dmapR(tar_idx) = dmapR(tar_idx) + ((cost(yP) - cost(yM)) ./ (2.*( 2.*cost(yD) - cost(yM) - cost(yP))));
    end
    %%%%%%%%%%
    
    dmapR=dmapR+dmin-1;
```

[matlab/imstereo_sch_stereo.m:38-42](../matlab/imstereo_sch_stereo.m#L38-L42)

MATLAB 인덱스는 1부터 시작하므로 `1 < idx <= D-1`이 곧 PyTorch의 "내부 인덱스"이고, 마지막에 `+dmin-1`로 disparity를 복원합니다. 평평한 곡선(분모 0)은 검사하지 않습니다.

### CUDA

```cpp
		if (sub_pixel)
		{
			float min_next_cost = (float)(costs[min(min_disp + 1, max_disp - 1) * img_h * img_w + idx]);
			float min_prev_cost = (float)(costs[max(min_disp - 1, 0) * img_h * img_w + idx]);
			float min_curr_cost = (float)(costs[min_disp * img_h * img_w + idx]);

			disp_map_float[idx] = (float)min_disp + (min_next_cost - min_prev_cost) / (2 * (2 * min_curr_cost - min_prev_cost - min_next_cost));
		}
```

[cuda/src/v1_baseline.cu:180-187](../cuda/src/v1_baseline.cu#L180-L187)

## 직접 해보기

```python
import sys; sys.path.insert(0, 'pytorch')
from torchvision.io import read_image, ImageReadMode
from census_stereo import census_transform, build_cost_volume, winner_take_all

L = read_image('data/sample/left.png', ImageReadMode.GRAY).unsqueeze(0)
R = read_image('data/sample/right.png', ImageReadMode.GRAY).unsqueeze(0)
cost = build_cost_volume(census_transform(L, 7), census_transform(R, 7), 0, 48, 11, +1)

y, x = 360, 640
i = cost[0, y, x].argmin().item()
cm, c0, cp = cost[0, y, x, i - 1:i + 2].tolist()
print('d* =', i, 'costs =', (cm, c0, cp))
print('manual offset =', (cp - cm) / (2 * (2 * c0 - cm - cp)))

d_int = winner_take_all(cost, 0, sub_pixel=False)
d_sub = winner_take_all(cost, 0, sub_pixel=True)
print('integer', d_int[0, y, x].item(), ' sub-pixel', d_sub[0, y, x].item())
print('changed pixels: %.3f' % ((d_sub != d_int).float().mean().item()))
print('argmin at 0 or 48: %.3f' % (((d_int == 0) | (d_int == 48)).float().mean().item()))
```

실행 결과는 다음과 같습니다.

```
d* = 5 costs = (1768.0, 1466.0, 1633.0)
manual offset = 0.1439232409381663
integer 5.0 sub-pixel 5.143923282623291
changed pixels: 0.935
argmin at 0 or 48: 0.061
```

수동 계산과 `winner_take_all`의 결과가 일치합니다. 약 93.5%의 픽셀이 보정되었고, 약 6.1%는 최솟값이 탐색 범위의 끝($d=0$ 또는 $48$)에 있어 PyTorch에서 보정이 생략됩니다. 끝 값이 많다면 `max_disp`가 너무 작거나 매칭이 실패한 영역이 많다는 신호입니다.

## 구현별 차이

| 항목 | MATLAB | PyTorch | CUDA |
|---|---|---|---|
| 보정 대상 | 내부 인덱스만 (`1 < idx <= D-1`) | 내부 인덱스만 (`0 < idx < N-1`) | 모든 픽셀 |
| 끝 인덱스 | 보정 없음 | 보정 없음 (정수 유지) | 이웃을 clamp하여 $\pm 0.5$ 보정이 생김 |
| 분모 0 (평평한 곡선) | 검사 없음 | 보정 생략 (`denom != 0`) | 검사 없음 (NaN/inf 가능) |
| 동점 처리 | `min`이 첫 최솟값 | `argmin` (보장 없음) | 엄격한 `>`라서 작은 disparity 우선 |
| 후보 개수 | `dmax-dmin+1` | `max_disp-min_disp+1` | `max_disp` (0 .. `max_disp-1`) |

CUDA에서 끝 인덱스는 clamp된 이웃이 중심과 같아져 보정량이 정확히 $-0.5$($d^*=0$) 또는 $+0.5$($d^*=D-1$)가 됩니다. 평평한 곡선의 NaN은 5장의 LR check에서 비교가 거짓이 되어 걸러지지 않을 수 있으므로 주의가 필요합니다. 이 구현 차이는 원본 코드를 수정하지 않은 상태 그대로의 기술입니다.

---

이전: [3장. 비용 집계](03_cost_aggregation.md) | 다음: [5장. 좌우 일관성 검사 (LR check)](05_lr_consistency.md)
