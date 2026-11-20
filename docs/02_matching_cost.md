# 2장. 매칭 비용: XOR + popcount (Hamming distance)

## 목표

- 두 census 비트열이 얼마나 다른지를 **Hamming distance**로 재고, 이를 XOR + popcount로 계산하는 방법을 배웁니다.
- 모든 후보 disparity에 대해 비용을 계산해 **cost volume**을 만드는 구조와 "shift 방향"의 의미를 이해합니다.
- 후보 disparity가 많을 때 픽셀 단위 비용만으로는 부족한 이유를 확인하고 3장으로 연결합니다.

## 개념 설명

Rectified 스테레오 쌍에서는 같은 3D 점이 두 영상의 **같은 행**에 나타나고, 열 위치만 달라집니다. 이 열 차이가 disparity $d$입니다. 따라서 왼쪽 픽셀 $(y, x)$에 대해 오른쪽 영상의 $(y, x-d)$, $d = d_{min}, \dots, d_{max}$를 후보로 두고 "누가 가장 비슷한가"를 비교하면 됩니다.

PyTorch 구현의 규약은 다음과 같습니다(모듈 docstring 참고).

- `disp_L[y, x] = d` 이면 left$(y,x)$가 right$(y, x-d)$와 대응합니다.
- `disp_R[y, x] = d` 이면 right$(y,x)$가 left$(y, x+d)$와 대응합니다.
- 두 값 모두 정상적인 영상쌍에서는 양수입니다(물체가 오른쪽 영상에서 왼쪽으로 이동하기 때문).

비용은 두 census 비트열의 **Hamming distance**(서로 다른 비트 개수)입니다. 비트열을 정수로 보면 `popcount(a XOR b)`입니다.

### 예시

1장의 3x3 예시와 가상의 후보 비트열을 비교합니다.

| 위치 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 왼쪽 $C_L$ | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| 후보 $C_R$ | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| XOR | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |

XOR에서 1이 2개이므로 cost는 2입니다. 비트열이 완전히 같으면 0, 완전히 반대이면 $k^2$입니다. 무작위로 다른 두 비트열의 기대값은 대략 $k^2/2$입니다.

## 수식

reference 영상 $A$, target 영상 $B$, 후보 disparity $d$에 대해 픽셀 단위 비용은 다음과 같습니다.

$$
c(y, x, d) = \operatorname{popcount}\big(C_A(y, x) \oplus C_B(y, x \mp d)\big) = \sum_{j=1}^{k^2} \big[ b^A_j(y,x) \ne b^B_j(y, x \mp d) \big]
$$

reference가 왼쪽이면 $x - d$, 오른쪽이면 $x + d$입니다. $d$를 $d_{min}$부터 $d_{max}$까지 바꾸어 쌓으면 $H \times W \times D$ 크기의 **cost volume**이 됩니다($D = d_{max} - d_{min} + 1$).

## 코드

### 영상을 $d$만큼 옮기기

```python
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
```

[pytorch/census_stereo.py:55-72](../pytorch/census_stereo.py#L55-L72)

`direction`은 "reference가 어느 쪽인가"를 뜻합니다. reference가 왼쪽이면(`+1`) target(오른쪽)을 `x-shift` 위치에서 가져옵니다. 이렇게 하면 모든 픽셀에서 `ref[x]`와 `tgt_shifted[x]`를 같은 위치끼리 XOR하면 됩니다. 소스가 없는 열은 0으로 채우며 wrap-around는 하지 않습니다.

### XOR + popcount

```python
    bz, h, w, _ = census_ref.shape
    n = max_disp - min_disp + 1
    box = torch.ones((1, 1, agg_size, agg_size), device=census_ref.device)
    volume = torch.empty((bz, h, w, n), dtype=torch.float32, device=census_ref.device)
    for i in range(n):
        tgt = shift_census(census_tgt, min_disp + i, direction)
        hamming = torch.bitwise_xor(census_ref, tgt).sum(dim=3, dtype=torch.float32)  # (B,H,W)
        volume[..., i] = F.conv2d(hamming.unsqueeze(1), box, padding='same')[:, 0]
    return volume
```

[pytorch/census_stereo.py:85-93](../pytorch/census_stereo.py#L85-L93)

`bitwise_xor`는 bool 텐서에서 원소별 XOR이고, 마지막 축(`K`)을 `sum`하면 비트가 1인 개수, 즉 popcount가 됩니다. bool을 비트 하나씩 저장했기 때문에 별도의 popcount 명령이 필요 없습니다. 결과 `hamming`은 $d$ 하나에 대한 `(B, H, W)` 픽셀 비용 맵입니다. 바로 다음 줄의 `conv2d`는 3장의 box filter 집계입니다.

### MATLAB

```matlab
    sch=ones([size(imR),dmax-dmin+1], class(imRC))*Inf;
    for d = 0:dmax-dmin
        imL_d = circshift(imLC,[0, -(dmin+d)]); % shift
        imL_d(:, end-(dmin+d):end, :) = 0;
        z = xor(imRC, imL_d);
        sch(:,:,d+1)= imfilter(sum(z, 3), ones(ws));
    end
```

[matlab/imstereo_sch_stereo.m:18-24](../matlab/imstereo_sch_stereo.m#L18-L24)

이 루프는 reference가 오른쪽인 경우입니다. `circshift`로 왼쪽 census를 옮기고(`imL_d(x) = imLC(x+s)`), 순환(wrap)되어 넘어온 열을 0으로 지운 뒤 `xor`와 `sum(z, 3)`(popcount)를 계산합니다. 구조는 PyTorch와 같습니다.

## 직접 해보기

```python
import sys; sys.path.insert(0, 'pytorch')
import torch
from torchvision.io import read_image, ImageReadMode
from census_stereo import census_transform, shift_census

L = read_image('data/sample/left.png', ImageReadMode.GRAY).unsqueeze(0)
R = read_image('data/sample/right.png', ImageReadMode.GRAY).unsqueeze(0)
cL, cR = census_transform(L, 7), census_transform(R, 7)

y, x = 420, 590
costs = []
for d in range(129):                     # reference = left
    tgt = shift_census(cR, d, +1)         # tgt[x] = cR[x - d]
    hamming = torch.bitwise_xor(cL, tgt).sum(dim=3)
    costs.append(hamming[0, y, x].item())
print(costs[60:69])
print('min cost', min(costs), 'at d =', costs.index(min(costs)))
print('cost at d=64:', costs[64], '/ candidates with cost <= that:', sum(c <= costs[64] for c in costs), 'of', len(costs))
```

실행 결과는 다음과 같습니다.

```
[29, 34, 35, 31, 15, 7, 13, 27, 27]
min cost 5 at d = 82
cost at d=64: 15 / candidates with cost <= that: 34 of 129
```

비용은 0~49 범위입니다. 첫 줄은 $d=60 \dots 68$의 비용으로, $d=65$에서 7까지 내려가는 골이 보입니다. 그러나 129개 후보 전체에서 최소는 $d=82$의 5이고, 이 픽셀의 픽셀 단위 최솟값은 이웃 픽셀들과 일관되지 않은 위치에서 나왔을 가능성이 높습니다(3장에서 집계하면 이 픽셀은 $d=64$ 부근으로 정리되고, 5장의 좌우 검사도 통과합니다). $d=64$의 비용 15보다 같거나 낮은 후보가 129개 중 34개나 있습니다. 샘플에는 능동 IR 도트 패턴이 있어 텍스처 자체는 풍부하지만, 49비트 하나만 비교하면 후보가 129개로 많은 만큼 우연히 낮은 비용이 나오는 후보도 늘어납니다. 즉 **픽셀 하나의 7x7 창(49비트)만으로는 이처럼 많은 후보 중에서 올바른 위치를 고르기에 판별력이 부족합니다.** 3장에서 같은 위치를 11x11 영역으로 집계하면 최소가 $d=64$에서 분명한 골을 이룹니다.

## 구현별 차이

| 항목 | MATLAB | PyTorch | CUDA |
|---|---|---|---|
| popcount | `sum(xor(...), 3)` (비트 단위) | `bitwise_xor(...).sum(dim=3)` (bool 단위) | 바이트 XOR 후 바이트별 비트 결합 (아래 참고) |
| shift 방식 | `circshift` 후 `d+1`열을 0으로 | 슬라이스 복사, 정확히 `d`열이 0 | 인덱스 `j+disp`를 영상 안으로 clamp |
| 비용 범위 | 0 ~ $k^2$ | 0 ~ $k^2$ | 의도는 0 ~ $k^2$ (아래 참고) |
| disparity 범위 | `dmax-dmin+1`개 | `max_disp-min_disp+1`개 | `0 .. max_disp-1` (`max_disp`개) |

MATLAB은 한 번 이동할 때마다 `d+1`개 열을 0으로 지우지만 PyTorch는 정확히 `d`개만 0입니다. MATLAB의 왼쪽 reference 루프(45~53행)는 shift 값이 음수라 `end-s:end` 범위가 비어 있어 순환된 열이 지워지지 않습니다. 영상 가장자리 $d$열에서만 차이가 나므로 결과 비교 시 가장자리는 제외하는 편이 좋습니다.

CUDA의 v1~v3 커널에는 확인이 필요한 부분이 있습니다(코드는 원본 그대로 둠). 바이트별 "popcount"가 비트를 `+`가 아닌 `|`로 합치도록 작성되어 있어, 코드대로라면 바이트당 0 또는 1만 기여합니다. 이 경우 비용은 *다른 비트 수*가 아니라 *다른 바이트 수*가 됩니다. 실제 사용에서는 결과가 정상적으로 나왔고 GPU에서 따로 검증한 내용은 아니므로, 문제일 수 있다는 정도로 참고하세요. census 패킹(1장)과 함께 [7장](07_cuda_baseline.md)에서 자세히 다룹니다.

---

이전: [1장. Census Transform](01_census_transform.md) | 다음: [3장. 비용 집계: box filter](03_cost_aggregation.md)
