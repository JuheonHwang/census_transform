# 5장. 좌우 일관성 검사 (LR check)

## 목표

- 왼쪽 기준과 오른쪽 기준 disparity를 서로 교차 검증해 신뢰할 수 없는 픽셀을 걸러내는 원리를 이해합니다.
- linear interpolation으로 다른 시점의 disparity를 **warp**하고 임계값과 비교하는 구현을 읽습니다.
- 임계값과 무효 픽셀 비율의 관계를 직접 확인합니다.

## 개념 설명

WTA는 각 픽셀이 독립적으로 최솟값을 고르므로, 가려진 영역(occlusion), 텍스처가 없는 영역, 반복 패턴에서는 틀린 disparity를 자신 있게 출력합니다. 이런 오류를 잡는 가장 간단한 방법이 **좌우 일관성 검사**입니다.

왼쪽 영상의 $(y,x)$가 오른쪽의 $(y, x-d_L)$과 대응한다고 판단했다면, 반대로 오른쪽 영상의 그 위치에서 출발한 매칭은 다시 $(y,x)$로 돌아와야 합니다. 즉 `disp_R`을 $x - d_L$에서 읽은 값이 $d_L$과 (거의) 같아야 합니다. 같지 않으면 둘 중 하나가 틀렸다는 뜻이므로 해당 픽셀을 **무효(0)** 로 처리합니다. 이 때문에 PyTorch 구현은 왼쪽/오른쪽 reference에서 각각 cost volume을 만들어 `disp_L`, `disp_R`을 모두 구합니다(`stereo_matching`).

규약을 다시 정리하면 다음과 같습니다(양수 disparity).

- `disp_L[y, x] = d` : left$(y,x)$ <-> right$(y, x-d)$
- `disp_R[y, x] = d` : right$(y,x)$ <-> left$(y, x+d)$

## 수식

sub-pixel disparity 때문에 $x - d_L(y,x)$는 정수가 아닙니다. 같은 행에서 선형 보간한 값을 $\tilde d_R$라 쓰겠습니다.

$$
\tilde d_R(y,x) = \operatorname{interp}_x\big(d_R(y,\cdot),\ x - d_L(y,x)\big), \qquad
\tilde d_L(y,x) = \operatorname{interp}_x\big(d_L(y,\cdot),\ x + d_R(y,x)\big)
$$

$$
d_L^{\text{out}}(y,x) = \begin{cases} d_L(y,x) & |\tilde d_R(y,x) - d_L(y,x)| \le \tau \\ 0 & \text{otherwise} \end{cases}
$$

$d_R$도 대칭적으로 $\tilde d_L$와 비교합니다. $\tau$는 임계값(threshold)이며 단위는 픽셀입니다.

### 수치 예시

샘플 영상의 $(y,x) = (420, 590)$(2~4장과 같은 픽셀)에서 `disp_L = 64.2259`입니다.

| 단계 | 값 |
|---|---|
| 샘플 위치 $x - d_L$ | $590 - 64.2259 = 525.7741$ |
| 이웃 열 | 525 (`disp_R` = 64.2516), 526 (`disp_R` = 64.3339) |
| 보간 가중치 $\alpha$ | 0.7741 |
| 보간 결과 | $0.7741 \times 64.3339 + 0.2259 \times 64.2516 \approx 64.3153$ |
| 차이 | $\lvert 64.3153 - 64.2259 \rvert \approx 0.089$ |

$\tau = 1$이므로 통과(유지)합니다. 반대로 가려진 픽셀에서 `disp_L = 100`인데 오른쪽에서 읽은 값이 94였다면 차이는 6이므로 0으로 지워집니다.

## 코드

### PyTorch

```python
def _sample_rows_linear(disp, cols):
    """Linearly sample disp (B, H, W) at fractional columns cols (B, H, W)."""
    w = disp.size(2)
    cols = cols.clamp(0, w - 1)
    c0 = torch.floor(cols).long()
    c1 = (c0 + 1).clamp(max=w - 1)
    alpha = cols - c0
    return alpha * disp.gather(2, c1) + (1 - alpha) * disp.gather(2, c0)
```

[pytorch/census_stereo.py:154-161](../pytorch/census_stereo.py#L154-L161)

열 좌표를 `[0, W-1]`로 clamp한 뒤, 바로 왼쪽 정수 열 `c0`와 오른쪽 열 `c1`의 값을 `alpha`로 섞습니다. `gather`는 픽셀마다 다른 열을 한 번에 읽습니다.

```python
def disparity_interp(disp_L, disp_R):
    """Warp each disparity map into the other view. Batch-correct.

    disp_L, disp_R: (B, H, W). Returns (disp_R2L, disp_L2R), both (B, H, W):
    disp_R2L[y, x] = disp_R sampled at x - disp_L[y, x] (linear, clamped);
    disp_L2R[y, x] = disp_L sampled at x + disp_R[y, x].
    """
    h, w = disp_L.shape[1:]
    x = torch.arange(w, device=disp_L.device, dtype=torch.float32).view(1, 1, w).expand_as(disp_L)
    return _sample_rows_linear(disp_R, x - disp_L), _sample_rows_linear(disp_L, x + disp_R)
```

[pytorch/census_stereo.py:164-173](../pytorch/census_stereo.py#L164-L173)

```python
def left_right_check(disp_L, disp_R, threshold):
    """Zero out pixels whose warped counterpart differs by more than threshold.

    disp_L, disp_R: (B, H, W). Returns new (disp_L, disp_R); inputs are unchanged.
    """
    disp_R2L, disp_L2R = disparity_interp(disp_L, disp_R)
    out_L = torch.where((disp_R2L - disp_L).abs() > threshold, torch.zeros_like(disp_L), disp_L)
    out_R = torch.where((disp_L2R - disp_R).abs() > threshold, torch.zeros_like(disp_R), disp_R)
    return out_L, out_R
```

[pytorch/census_stereo.py:176-184](../pytorch/census_stereo.py#L176-L184)

`x`는 열 인덱스 격자 `0..W-1`입니다. `x - disp_L`이 왼쪽 픽셀이 오른쪽 영상에서 가리키는 열이고, 거기서 `disp_R`을 읽은 것이 `disp_R2L`입니다. 임계값을 넘으면 `torch.where`로 0을 넣습니다. 입력을 직접 고치지 않고 새 텐서를 반환합니다. 이 함수까지 호출하는 전체 파이프라인은 [6장](06_pytorch_walkthrough.md)의 `get_disparity`입니다.

### MATLAB

```matlab
        [H, W] = size(dmapL);
        [X,Y] = meshgrid(G(1:W), G(1:H));
        dmapRtoL = interp2(dmapR, X-dmapL, Y, 'linear', 0);
        dmapLtoR = interp2(dmapL, X+dmapR, Y, 'linear', 0);
        
        dmapL(abs(dmapRtoL - dmapL) > thres_LR_disp) = 0;
        dmapR(abs(dmapLtoR - dmapR) > thres_LR_disp) = 0;
```

[matlab/matching.m:52-58](../matlab/matching.m#L52-L58)

같은 논리이며 `interp2`가 선형 보간을 합니다. 마지막 인자 `0`은 영상 밖을 샘플링하면 0을 반환한다는 뜻입니다.

### CUDA

CUDA는 두 커널로 나눕니다. `GetDispInterp`가 보간된 값을 만들고(열 좌표를 영상 안으로 clamp), `ThresholdDisp`가 비교 후 0으로 지웁니다.

```cpp
		if (abs(interp_disp_map[idx] - disp_map[idx]) > threshold)
		{
			disp_map[idx] = 0;
		}
```

[cuda/src/v1_baseline.cu:223-226](../cuda/src/v1_baseline.cu#L223-L226)

## 직접 해보기

전체 해상도에서 CPU로 수십 초 걸립니다(측정 환경(노트북 CPU)에서는 매칭 단계가 약 21초).

```python
import sys, time; sys.path.insert(0, 'pytorch')
import torch
from torchvision.io import read_image, ImageReadMode
from census_stereo import stereo_matching, left_right_check, disparity_interp

L = read_image('data/sample/left.png', ImageReadMode.GRAY).unsqueeze(0)
R = read_image('data/sample/right.png', ImageReadMode.GRAY).unsqueeze(0)
t = time.time()
dL, dR = stereo_matching(L, R, 0, 128, 11, 7, True, 'cpu')
print('matching: %.0f s' % (time.time() - t))

for tau in (1, 3, 10):
    oL, oR = left_right_check(dL, dR, tau)
    print('threshold %2d: invalid L %.1f%%  R %.1f%%' %
          (tau, 100 * (oL == 0).float().mean(), 100 * (oR == 0).float().mean()))

oL, oR = left_right_check(dL, dR, 1)
valid = oL > 0
print('valid disparity median %.2f, max %.2f' % (oL[valid].median(), oL.max()))

# 수치 예시와 같은 픽셀
y, x = 420, 590
R2L, _ = disparity_interp(dL, dR)
xr = x - dL[0, y, x].item()
print('disp_L = %.4f, sample column = %.4f' % (dL[0, y, x], xr))
c = int(xr)
print('disp_R[%d] = %.4f, disp_R[%d] = %.4f' % (c, dR[0, y, c], c + 1, dR[0, y, c + 1]))
print('interp = %.4f, diff = %.3f' % (R2L[0, y, x], (R2L[0, y, x] - dL[0, y, x]).abs()))
```

실행 결과는 다음과 같습니다.

```
matching: 21 s
threshold  1: invalid L 10.0%  R 10.2%
threshold  3: invalid L 9.6%  R 9.9%
threshold 10: invalid L 8.3%  R 8.6%
valid disparity median 56.58, max 128.00
disp_L = 64.2259, sample column = 525.7741
disp_R[525] = 64.2516, disp_R[526] = 64.3339
interp = 64.3153, diff = 0.089
```

임계값을 줄이면 무효 픽셀이 늘고 남은 픽셀의 신뢰도는 높아집니다. 이 샘플에서는 $\tau=1$에서 약 90%의 픽셀이 유효하며(무효 약 10%), 임계값을 10까지 완화해도 무효 비율이 8%대에 머무릅니다. 즉 일관성 검사에서 걸러지는 픽셀의 대부분은 임계값을 아무리 키워도 통과하지 못하는 큰 불일치입니다. 또한 유효한 disparity의 중앙값이 약 56.6 px이고 `max_disp=128`의 끝값(128)도 일부 남아 있으므로, 결과를 쓸 때 탐색 범위 끝 근처 값은 의심해 볼 만합니다. 무효 픽셀의 주된 원인은 일반적으로 (1) 왼쪽 영상 가장자리에서 오른쪽에 대응 점이 없는 영역, (2) 가려진 영역, (3) 텍스처가 부족한 영역입니다. 결과를 이미지로 저장해 보려면 `save_disparity(oL, oR, 'out')`를 사용합니다.

## 구현별 차이

| 항목 | MATLAB | PyTorch | CUDA |
|---|---|---|---|
| 보간 | `interp2(..., 'linear', 0)` | `gather` 두 번 + 선형 가중 | 두 이웃 열의 선형 가중 |
| 영상 밖 샘플 | 0을 반환 | 열 좌표를 `[0, W-1]`로 clamp | 열 좌표를 `[0, W-1]`로 clamp |
| 무효 표시 | 0 | 0 | 0 |
| 임계값 | 실수 (`thres_LR_disp`, 기본 10) | 실수 (`threshold`, 함수 기본 10, `--lr-threshold` 기본 10) | 정수 (`const int threshold`) |
| 비교 조건 | `abs(diff) > thr`이면 0 | 동일 | 동일 |

MATLAB은 영상 밖에서 0을 돌려주므로 차이가 크게 나 무효가 되지만, PyTorch/CUDA는 가장자리 값을 복제해 쓰므로 가장자리 픽셀이 더 쉽게 통과할 수 있습니다. 비교 조건은 세 구현 모두 `>`이므로 차이가 정확히 임계값과 같으면 유지됩니다.

CUDA 구현은 disparity를 `0 .. max_disp-1`에서 찾고, 첫 번째 입력 영상(코드상 "left")의 census를 두 번째 영상의 `x + disp` 위치와 비교합니다. 이 방향은 위의 PyTorch 규약(left$(x)$ <-> right$(x-d)$)과 반대이므로, CUDA 출력의 left/right가 어느 영상에 대응하는지는 [7장](07_cuda_baseline.md)에서 확인하기 바랍니다.

---

이전: [4장. Disparity 선택](04_disparity_subpixel.md) | 다음: [6장. PyTorch 코드 따라가기](06_pytorch_walkthrough.md) (이후 [7장 CUDA baseline](07_cuda_baseline.md), [8장 kernel fusion](08_cuda_kernel_fusion.md), [9장 rolling buffer](09_cuda_rolling_buffer.md))
