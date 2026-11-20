# 1장. Census Transform

## 목표

- 이웃 픽셀과 중심 픽셀의 **밝기 대소 관계**만으로 픽셀을 설명하는 census transform을 이해합니다.
- IR(근적외선) 스테레오 영상에 census transform이 잘 맞는 이유를 설명할 수 있습니다.
- `census_transform()`의 `unfold` 기반 구현과 MATLAB/CUDA 구현의 차이를 파악합니다.

## 개념 설명

스테레오 매칭은 왼쪽 영상의 한 픽셀이 오른쪽 영상의 어느 픽셀과 같은 3D 점인지 찾는 문제입니다. 가장 단순한 방법은 밝기 값의 차이(SAD 등)를 비교하는 것이지만, 두 카메라의 밝기 응답이 다르면 같은 점이라도 밝기 값이 달라져 실패합니다.

Census transform(Zabih and Woodfill, "Non-parametric local transforms for computing visual correspondence", ECCV 1994)은 밝기 값 자체를 버리고, 각 픽셀 주변 $k \times k$ 창에서 "중심보다 밝은가?"라는 1비트 질문의 답만 모아 $k^2$ 비트 문자열로 바꿉니다. 이후 두 픽셀의 유사도는 이 비트 문자열이 얼마나 다른지(Hamming distance, 2장)로 측정합니다.

### IR 영상에 잘 맞는 이유

IR 스테레오(예: 능동 패턴 프로젝터를 쓰는 카메라)에서는 다음 문제가 흔합니다.

- 좌우 IR 센서의 **gain/exposure 차이**로 같은 점의 밝기 값이 서로 다릅니다.
- 프로젝터 조명이 **불균일**(중심이 밝고 가장자리가 어두움)하여 영상 전체에 느린 밝기 변화가 있습니다.
- 거리에 따라 반사광 세기가 달라져 국소적인 밝기 스케일이 변합니다.

Census transform은 창 안의 밝기를 **단조 증가 변환**(monotonic transform, 예: $I' = aI + b,\ a>0$ 또는 gamma)해도 결과가 바뀌지 않습니다. 대소 관계가 유지되기 때문입니다. 따라서 위의 gain/exposure 차이와 완만한 조명 불균일(창 크기 안에서는 거의 단조 변환으로 근사됨)에 강합니다.

한편 census는 **국소적인 밝기 순서**에 의존하므로, 창 안에 순서를 구분할 만한 텍스처가 있어야 합니다. IR 도트 패턴은 바로 이런 텍스처를 인위적으로 만들어 주므로 궁합이 좋습니다. 이 튜토리얼의 샘플 영상(Occipital Structure Core로 촬영한 능동 IR 스테레오 쌍, 출처는 [data/README.md](../data/README.md))에서도 두 영상 모두 책상 위 물체와 벽면 위에 조밀한 점·반점(speckle) 패턴이 투영되어 있는 것을 볼 수 있습니다. 반대로 완전히 평평한 영역에서는 비트가 노이즈에 의해 결정되어 매칭이 불안정합니다.

## 수식

영상 $I$, 홀수 창 크기 $k$, 반경 $r = \lfloor k/2 \rfloor$일 때 픽셀 $p=(x,y)$의 census 비트는 다음과 같습니다.

$$
b_{(u,v)}(p) = \begin{cases} 1 & I(x+u,\, y+v) > I(x,y) \\ 0 & \text{otherwise} \end{cases}, \qquad -r \le u,v \le r
$$

$C(p)$는 이 $k^2$개의 비트를 정해진 순서(행 우선)로 이어 붙인 문자열입니다. 부등호가 **엄격**(strictly)하므로 같은 값은 0이며, 중심 비트는 항상 0입니다.

### 3x3 예시 (실제 숫자)

| 52 | 60 | 48 |
|:--:|:--:|:--:|
| 55 | **58** | 70 |
| 40 | 58 | 61 |

중심 값은 58입니다. 각 이웃이 58보다 큰지 비교합니다.

| 0 (52) | 1 (60) | 0 (48) |
|:--:|:--:|:--:|
| 0 (55) | - (중심) | 1 (70) |
| 0 (40) | 0 (58, 같음) | 1 (61) |

행 우선으로 이으면 `0 1 0 0 0 1 0 0 1`이고, 가운데(인덱스 4)는 항상 0입니다. 아래쪽 가운데 픽셀은 58로 중심과 같아서 엄격한 부등호 때문에 0입니다. 이 입력에 `x*0.6 + 10`을 적용해도 비트열은 같습니다.

## 코드

### PyTorch

```python
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
```

[pytorch/census_stereo.py:31-49](../pytorch/census_stereo.py#L31-L49)

동작 순서는 다음과 같습니다.

1. `F.pad`로 상하좌우에 $r$만큼 **0**을 채웁니다. 경계 픽셀은 "밝기 0인 가상 이웃"과 비교됩니다.
2. `F.unfold`가 모든 위치의 $k \times k$ 창을 열로 펼쳐 `(B, k*k, H*W)` 텐서를 만듭니다. 한 열이 한 픽셀의 창입니다.
3. 가운데 행(`centre`)을 중심값으로 삼아 `blocks > center`를 한 번에 비교합니다. 브로드캐스팅으로 모든 픽셀이 동시에 처리됩니다.
4. 결과를 `(B, H, W, k*k)` bool 텐서로 되돌립니다. 픽셀당 $k^2$개의 bool(바이트)을 저장하므로 메모리는 많이 쓰지만 이해하기 쉽습니다.

### MATLAB

```matlab
function imLCensus = cenesus(imL, wc)
    imLmargin = zeros(size(imL)+[wc wc]-1, class(imL));
    imLmargin(floor((wc-1)/2)+(1:size(imL, 1)),floor((wc-1)/2)+(1:size(imL, 2))) = imL;
    imLBlock = im2col(imLmargin, [wc wc], 'sliding');
    imLBlock = imLBlock > imLBlock(ceil((wc*wc)/2), :);
    %imLBlock(ceil((wc*wc)/2), :) = [];
    imLCensus = permute(reshape(imLBlock, [wc*wc size(imL)]), [2 3 1]);
end
```

[matlab/imstereo_sch_stereo.m:74-81](../matlab/imstereo_sch_stereo.m#L74-L81)

구조가 같습니다. `im2col(..., 'sliding')`이 `unfold`, `ceil((wc*wc)/2)`(1-based)가 `centre`(0-based)에 해당합니다. 주석 처리된 줄은 중심 비트를 제거하던 옵션이며 현재는 쓰지 않습니다.

### CUDA

CUDA 커널(`VectorCensusTransform`)은 한 스레드가 (픽셀, 바이트) 하나를 맡아 최대 8비트를 `census <<= 1; census |= (image_center < neighbour)`로 조립합니다. 경계에서는 좌표를 영상 안으로 clamp합니다. [cuda/src/v1_baseline.cu:49-78](../cuda/src/v1_baseline.cu#L49-L78)

## 직접 해보기

저장소 루트에서 실행합니다.

```python
import sys; sys.path.insert(0, 'pytorch')
import torch
from torchvision.io import read_image, ImageReadMode
from census_stereo import census_transform

# 1) 위 3x3 예시
p = torch.tensor([[52., 60, 48], [55, 58, 70], [40, 58, 61]]).reshape(1, 1, 3, 3)
print(census_transform(p, 3)[0, 1, 1].int().tolist())   # [0, 1, 0, 0, 0, 1, 0, 0, 1]

# 2) 단조 변환 불변성 (float로 변환해 양자화 영향을 배제)
L = read_image('data/sample/left.png', ImageReadMode.GRAY).unsqueeze(0)
c0 = census_transform(L, 7)
print(c0.shape, c0.dtype)                                          # (1, 896, 1216, 49) bool
print((c0 == census_transform(L.float() * 0.6 + 10, 7)).float().mean().item())  # gain/offset
print((c0 == census_transform(L.float() ** 1.5, 7)).float().mean().item())      # gamma

# 3) 좌우 영상은 같은 좌표에서는 다른 비트를 가진다 (시차 때문)
R = read_image('data/sample/right.png', ImageReadMode.GRAY).unsqueeze(0)
print((c0 == census_transform(R, 7)).float().mean().item())
```

실행 결과는 다음과 같습니다.

```
[0, 1, 0, 0, 0, 1, 0, 0, 1]
torch.Size([1, 896, 1216, 49]) torch.bool
1.0
1.0
0.5120044350624084
```

gain/offset과 gamma를 적용해도 비트가 100% 같습니다. 반면 좌우 영상을 같은 좌표에서 비교하면 약 51%만 일치합니다. 샘플의 disparity가 수십 픽셀(대략 40~100 px)로 크기 때문에 같은 좌표의 두 창은 거의 서로 다른 장면 조각을 보고 있고, 일치율이 무작위 비트의 기대값(50%)에 가깝습니다. 시차만큼 어긋나 있기 때문이며, 어긋남을 찾아내는 것이 다음 장의 일입니다.

## 구현별 차이

| 항목 | MATLAB | PyTorch | CUDA |
|---|---|---|---|
| 비트 정의 | 이웃 > 중심 | 이웃 > 중심 | 이웃 > 중심 (`center < neighbour`) |
| 경계 처리 | 0으로 padding | 0으로 padding (`F.pad`) | 좌표를 영상 안으로 clamp (edge 복제) |
| 저장 형태 | `(H, W, k*k)` logical | `(B, H, W, k*k)` bool, 픽셀당 $k^2$ 바이트 | 8비트씩 bit-packing한 uchar |
| 중심 비트 | 포함 (항상 0) | 포함 (항상 0) | 포함 (항상 0) |

경계 처리가 다르므로 영상 가장자리 $r$ 픽셀 폭에서는 세 구현의 결과가 일치하지 않습니다. 내부 영역은 같습니다.

CUDA 구현에는 확인이 필요한 부분이 두 가지 있습니다(코드는 원본 그대로 두었으며 7장에서 다룹니다). 하나는 census 버퍼가 픽셀당 `ceil(K*K/8)` 바이트인데 행은 `ceil(K/8)` 바이트씩 패킹되어, 코드를 읽어 보면 $K=11$일 때 창의 0~10행 중 0~7행만 인코딩되는 것으로 보인다는 점입니다. 다른 하나는 2장에서 설명하는 popcount 부분입니다. 실제 사용에서는 결과가 정상적으로 나왔고 GPU에서 따로 검증한 내용은 아니므로, 문제일 수 있다는 정도로 참고하세요. 자세한 내용은 [7장](07_cuda_baseline.md)에 있습니다.

---

이전: 없음(첫 장) | 다음: [2장. 매칭 비용: XOR + popcount](02_matching_cost.md)
