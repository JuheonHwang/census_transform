# 09. Rolling buffer: max_disp와 무관한 메모리

[08장](08_cuda_kernel_fusion.md)에서 XOR buffer와 중복된 volume을 없앴지만, cost volume `4DHW` byte는 그대로 남았습니다. 이 장의 v3([cuda/src/v3_rolling.cu](../cuda/src/v3_rolling.cu))는 volume 자체를 없앱니다. 빌드와 실행은 07장과 같고 실행 파일은 `census_v3`입니다. 이 장이 마지막이며, 앞의 장들과 마찬가지로 이 코드는 컴파일하거나 실행해 본 적이 없습니다.

## 1. 아이디어: 최솟값 주변의 세 값이면 충분합니다

WTA는 픽셀마다 `cost(d)`가 최소인 `d*`를 고릅니다. sub-pixel은 거기에 `cost(d*-1)`, `cost(d*)`, `cost(d*+1)` 세 값으로 포물선을 맞춥니다 ([04장](04_disparity_subpixel.md)). 그 밖의 `cost(d)`는 결과에 쓰이지 않습니다. 그러므로 disparity를 0부터 D-1까지 **순서대로** 처리하면서 다음 두 가지만 유지하면 같은 결과를 얻습니다.

- 최근 세 disparity의 cost를 담은 **3-slice 창**(`cost(d-2)`, `cost(d-1)`, `cost(d)`)
- 픽셀마다 지금까지의 **최솟값 기록**: 최소였던 disparity `d*`와 그때의 세 cost `(cost(d*-1), cost(d*), cost(d*+1))`

메모리는 D가 아니라 slice 개수(상수)에 비례하게 됩니다.

## 2. 구현된 rolling scheme

### buffer와 초기화

v3가 할당하는 것은 3-slice buffer 두 개(`Cost_a`, `Cost_b`), 최솟값 기록용 `minCosts`(slice 3개)와 `minDispValue`입니다 ([cuda/src/v3_rolling.cu:324-335](../cuda/src/v3_rolling.cu#L324-L335)).

[cuda/src/v3_rolling.cu:324-335](../cuda/src/v3_rolling.cu#L324-L335)

```cuda
	// CensusSum cudaMalloc
	checkCudaErrors(cudaMalloc(&d_CensusSum_mem, img_height * img_width * sizeof(ushort)));

	// Costs cudaMalloc
	checkCudaErrors(cudaMalloc(&d_Cost_a_mem, 3 * img_height * img_width * sizeof(uint)));
	checkCudaErrors(cudaMalloc(&d_Cost_b_mem, 3 * img_height * img_width * sizeof(uint)));

	checkCudaErrors(cudaMalloc(&d_minCosts_mem, 3 * img_height * img_width * sizeof(uint)));
	// cudaMemset fills BYTES: UINT_MAX is truncated to 0xFF, so every uint becomes 0xFFFFFFFF (= UINT_MAX)
	checkCudaErrors(cudaMemset(d_minCosts_mem, UINT_MAX, 3 * img_height * img_width * sizeof(uint)));

	checkCudaErrors(cudaMalloc(&d_minDispValue_mem, img_height * img_width * sizeof(ushort)));
```

한 slice는 `H*W`개의 uint이고 `s`번째 slice의 픽셀 `idx`는 `buffer[s * H * W + idx]`입니다. 두 buffer는 반복마다 역할(`curr`, `prev`)을 바꿉니다 (아래 `VectorShift` 참고).

`cudaMemset(ptr, value, count)`는 `value`를 **byte** 단위로 채웁니다. `UINT_MAX`는 `unsigned char`로 줄어들어 `0xFF`가 되고, 따라서 각 uint가 `0xFFFFFFFF`(= `UINT_MAX`)가 됩니다. "아직 아무 후보도 없음"을 나타내는 가장 큰 값이므로, 첫 후보는 항상 `<` 비교에서 이깁니다. right pass를 시작하기 전에도 같은 초기화를 다시 합니다 ([cuda/src/v3_rolling.cu:373](../cuda/src/v3_rolling.cu#L373)). `minDispValue`는 초기화하지 않아도 되는데, 첫 후보가 항상 기록되기 때문입니다.

### disp = 0 단계

루프에 들어가기 전에 disparity 0을 한 번 처리합니다 ([cuda/src/v3_rolling.cu:337-345](../cuda/src/v3_rolling.cu#L337-L345)).

[cuda/src/v3_rolling.cu:337-345](../cuda/src/v3_rolling.cu#L337-L345)

```cuda
	// left
	// initialize
	VectorCensusXOR_N_Sum <<<kNumBlocks, kNumThreads>>> (d_CensusSum_mem, d_leftCensus_mem, d_rightCensus_mem, census_size, census_filter_size, img_height, img_width, 0);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	VectorBox_N_Cost <<<kNumBlocks, kNumThreads>>> (d_minCosts_mem, d_minDispValue_mem, d_Cost_a_mem, d_CensusSum_mem, aggreate_filter_size, img_height, img_width, 0);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
```

`VectorBox_N_Cost`는 `disp == 0`이면 cost(0)을 slice 1과 2에 모두 넣습니다. 다음 반복에서 shift가 일어나면 slice 0, 1이 둘 다 cost(0)이 되는데, 이것이 `cost(-1)`을 `cost(0)`으로 clamp한 값입니다 (v1/v2의 `max(min_disp - 1, 0)`과 같은 효과).

### 한 번의 반복: Shift, XOR_N_Sum, Box_N_Cost

`disp = 1 .. D` 반복은 다음과 같습니다 ([cuda/src/v3_rolling.cu:348-361](../cuda/src/v3_rolling.cu#L348-L361)).

[cuda/src/v3_rolling.cu:348-361](../cuda/src/v3_rolling.cu#L348-L361)

```cuda
	for (int disp = 1; disp < (max_disp + 1); disp++) {
		// shift
		VectorShift <<<kNumBlocks, kNumThreads>>> (((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), ((disp % 2) ? d_Cost_a_mem : d_Cost_b_mem), img_height, img_width);
        cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
		
		VectorCensusXOR_N_Sum <<<kNumBlocks, kNumThreads>>> (d_CensusSum_mem, d_leftCensus_mem, d_rightCensus_mem, census_size, census_filter_size, img_height, img_width, min(disp, max_disp - 1));
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorBox_N_Cost <<<kNumBlocks, kNumThreads>>> (d_minCosts_mem, d_minDispValue_mem, ((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), d_CensusSum_mem, aggreate_filter_size, img_height, img_width, disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
	}
```

세 kernel의 역할입니다.

**(1) `VectorShift(curr, prev)`.** 창을 한 칸 밉니다. `curr[0] = prev[1]`, `curr[1] = prev[2]`입니다 ([cuda/src/v3_rolling.cu:180-186](../cuda/src/v3_rolling.cu#L180-L186)).

[cuda/src/v3_rolling.cu:180-186](../cuda/src/v3_rolling.cu#L180-L186)

```cuda
__global__ void VectorShift(uint* cost_curr, uint* cost_prev, const int img_h, const int img_w)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		cost_curr[0 * img_h * img_w + idx] = cost_prev[1 * img_h * img_w + idx];
		cost_curr[1 * img_h * img_w + idx] = cost_prev[2 * img_h * img_w + idx];
	}
}
```

`disp`가 홀수면 `curr = Cost_b`, `prev = Cost_a`이고 짝수면 반대입니다. disp = 0은 `Cost_a`에 썼으므로 disp = 1에서는 `Cost_a -> Cost_b`로 밀립니다. 각 thread는 자기 `idx`만 만지므로 제자리 shift도 가능해 보이지만, 코드는 읽는 buffer와 쓰는 buffer를 나누는 방식을 택했습니다.

**(2) `VectorCensusXOR_N_Sum`.** 08장의 fused kernel로, `disp`의 픽셀별 cost를 `CensusSum`에 씁니다. 인자는 `min(disp, max_disp - 1)`입니다 (아래 참고).

**(3) `VectorBox_N_Cost`.** box filter로 `cost(disp)`를 구해 `curr[2]`에 넣고, 최솟값 기록을 갱신합니다 ([cuda/src/v3_rolling.cu:155-172](../cuda/src/v3_rolling.cu#L155-L172)). box 합산 부분(144-153행)은 `VectorBox`와 똑같습니다.

[cuda/src/v3_rolling.cu:155-172](../cuda/src/v3_rolling.cu#L155-L172)

```cuda
		if (disp == 0)
		{
			cost_dummy[1 * img_h * img_w + idx] = cost_summation;
			cost_dummy[2 * img_h * img_w + idx] = cost_summation;
		}
		else
		{
			cost_dummy[2 * img_h * img_w + idx] = cost_summation;

			if (cost_dummy[1 * img_h * img_w + idx] < minCosts[1 * img_h * img_w + idx])
			{
				minDispValue[idx] = (disp - 1);

				minCosts[0 * img_h * img_w + idx] = cost_dummy[0 * img_h * img_w + idx];
				minCosts[1 * img_h * img_w + idx] = cost_dummy[1 * img_h * img_w + idx];
				minCosts[2 * img_h * img_w + idx] = cost_dummy[2 * img_h * img_w + idx];
			}
		}
```

`curr`의 의미는 `[0] = cost(d-2)`, `[1] = cost(d-1)`, `[2] = cost(d)`입니다 (d = `disp`). 갱신 조건은 `cost(d-1)`, 즉 **가운데 slice**가 지금까지의 최솟값 `minCosts[1]`보다 **엄격히** 작은지입니다. 이기면 `minDispValue = d - 1`이고 세 slice를 통째로 `minCosts`로 복사합니다. 후보 `d-1`을 한 단계 늦게 평가하는 이유는, 그 후보의 "다음 이웃" `cost(d)`가 이제야 준비되기 때문입니다.

### 왜 `max_disp`까지 포함해 도는가, 그리고 `min(disp, max_disp - 1)`

후보 `d-1`은 반복 `d`에서 평가되므로, 마지막 후보 `D-1`을 평가하려면 `disp = D`까지 돌아야 합니다. 그래서 루프는 `disp < max_disp + 1`, 총 `D + 1`번 반복입니다. 그런데 반복 `D`에서 cost(D)를 실제로 계산하면 탐색 범위 밖 disparity의 값이 "next"로 들어갑니다. v1/v2는 `D-1`에서 next 이웃을 `min(min_disp + 1, max_disp - 1)`로 clamp해 cost(D-1)을 복제하므로 ([v1:182](../cuda/src/v1_baseline.cu#L182)), v3도 XOR 인자를 `min(disp, max_disp - 1)`로 잘라 같은 결과(cost(D) := cost(D-1))를 만듭니다. 소스 주석도 같은 설명입니다 ([cuda/src/v3_rolling.cu:130-131](../cuda/src/v3_rolling.cu#L130-L131)). right pass는 `-min(disp, max_disp - 1)`입니다 ([389행](../cuda/src/v3_rolling.cu#L389)).

### 한 픽셀 추적

`D = 5`이고 한 픽셀의 box 후 cost가 `cost(0..4) = 7, 5, 3, 4, 6`이라고 합시다. 반복 `disp = 0 .. 5`에서 `curr`의 세 slice(box 직후), 가운데 slice 검사, 최솟값 기록은 다음과 같습니다. `Cost_a`, `Cost_b`는 A, B로 줄여 씁니다. `-`는 쓰이지 않는 값입니다.

| disp | curr | XOR 인자 | `[0]` | `[1]` | `[2]` | 검사: `[1] < minCosts[1]` | minDispValue | minCosts (`[0],[1],[2]`) |
|---|---|---|---|---|---|---|---|---|
| (초기) | | | | | | | - | (inf, inf, inf) |
| 0 | A | 0 | - | 7 | 7 | 없음 (disp == 0) | - | (inf, inf, inf) |
| 1 | B | 1 | 7 | 7 | 5 | 7 < inf, 갱신 | 0 | (7, 7, 5) |
| 2 | A | 2 | 7 | 5 | 3 | 5 < 7, 갱신 | 1 | (7, 5, 3) |
| 3 | B | 3 | 5 | 3 | 4 | 3 < 5, 갱신 | 2 | (5, 3, 4) |
| 4 | A | 4 | 3 | 4 | 6 | 4 < 3 아님 | 2 | (5, 3, 4) |
| 5 | B | min(5, 4) = 4 | 4 | 6 | 6 | 6 < 3 아님 | 2 | (5, 3, 4) |

`disp = 1`의 `[0] = 7`은 clamp된 cost(-1)이고, `disp = 5`의 `[2] = 6`은 복제된 cost(4)입니다. 결과는 `d* = 2`, `(prev, curr, next) = (5, 3, 4)`이고, sub-pixel은 `2 + (4 - 5) / (2 * (2*3 - 5 - 4)) = 2 + (-1) / (-6) = 2.1667`입니다. v1의 `GetDispFloat`가 전체 volume에서 `d* = 2`, `prev = cost(1) = 5`, `next = cost(3) = 4`를 읽어 얻는 값과 같습니다. 최솟값이 `D-1`에 있으면 마지막 반복에서 `(cost(D-2), cost(D-1), cost(D-1))`이 기록되어 v1/v2와 마찬가지로 +0.5 오프셋이 됩니다 ([확인이 필요한 부분 3](07_cuda_baseline.md#확인이-필요한-부분)).

## 3. v3의 `GetDispFloat`: 스캔이 없습니다

winner는 이미 기록되어 있으므로 kernel은 기록을 읽어 식에 넣기만 합니다 ([cuda/src/v3_rolling.cu:197-210](../cuda/src/v3_rolling.cu#L197-L210)).

[cuda/src/v3_rolling.cu:197-210](../cuda/src/v3_rolling.cu#L197-L210)

```cuda
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		if (sub_pixel)
		{
			float min_prev_cost = (float)(minCosts[0 * img_h * img_w + idx]);
			float min_curr_cost = (float)(minCosts[1 * img_h * img_w + idx]);
			float min_next_cost = (float)(minCosts[2 * img_h * img_w + idx]);

			disp_map_float[idx] = (float)minDispValue[idx] + (min_next_cost - min_prev_cost) / (2 * (2 * min_curr_cost - min_prev_cost - min_next_cost));
		}
		else
		{
			disp_map_float[idx] = (float)minDispValue[idx];
		}
	}
```

v1/v2의 `for (disp = 0; disp < max_disp; ...)` 스캔과 volume 읽기가 사라졌습니다 (`max_disp` 인자는 받지만 본문에서 쓰지 않습니다). 식과 확인이 필요한 부분(분모 0, 범위 끝 +-0.5)은 v1과 같습니다.

## 4. 메모리: D와 무관합니다

v3의 device buffer ([cuda/src/v3_rolling.cu:1-11](../cuda/src/v3_rolling.cu#L1-L11) 주석과 host 코드 기준)입니다.

| buffer | 크기 (byte) |
|---|---|
| census x2 | 2nHW |
| CensusSum (ushort) | 2HW |
| Cost_a, Cost_b (uint x 3 slice 각각) | 2 x 12HW = 24HW |
| minCosts (uint x 3 slice) | 12HW |
| minDispValue (ushort) | 2HW |
| left float disparity (right pass 중 존재) | 4HW |

peak는 right pass 중이고 D가 식에 나오지 않습니다.

> peak(v3) = 2nHW + 2HW + 24HW + 12HW + 2HW + 4HW = **2nHW + 44HW**

right pass 이후 단계(float map 2개, interpolation 2개, uchar 2개)는 이보다 작습니다. header 주석은 left float map(`4HW`)을 뺀 `2nHW + 40HW`를 "about"으로 적었고, 이 문서는 할당 순서를 따라 `4HW`를 포함합니다 (1280 x 720에서 3.5 MiB 차이).

### v1 / v2 / v3 비교

1280 x 720은 HW = 921,600, 샘플 1216 x 896은 HW = 1,089,536이며, MiB = 2^20 byte. 방향당 launch는 한 방향의 disparity 루프 launch 수이고, 전체는 census, LR check, uchar 변환(각 2, 2, 2, 2)을 포함합니다.

| | v1 | v2 | v3 |
|---|---|---|---|
| kernel 종류 | 8 | 7 | 8 (VectorShift, VectorBox_N_Cost 추가) |
| 방향당 launch | 3D | 2D | 3D + 2 |
| 전체 launch | 6D + 10 | 4D + 10 | 6D + 14 |
| peak 메모리 (byte) | 3nHW + 2HW + 8DHW | 2nHW + 6HW + 4DHW | 2nHW + 44HW |
| 1280x720, K = 11, D = 300 (n = 16) | 2153.3 MiB | 1088.1 MiB | 66.8 MiB |
| 샘플 1216x896, K = 7, D = 128 (n = 7) | 1087.9 MiB | 552.8 MiB | 60.3 MiB |
| D = 300 전체 launch (해상도 무관) | 1810 | 1210 | 1814 |
| D = 128 전체 launch (해상도 무관) | 778 | 522 | 782 |
| 실행 시간 | 측정 필요 | 측정 필요 | 측정 필요 |

v3의 메모리는 K = 11, D = 300에서 v2의 약 1/16(66.8 / 1088.1), 샘플 설정(K = 7, D = 128)에서 약 1/9.2(60.3 / 552.8)입니다. D가 커질수록 격차는 커집니다. 한편 launch 횟수는 v3가 v2보다 **많습니다**. `VectorShift`가 새로 생겼고 disp = 0 단계가 루프 밖에 한 번 더 있기 때문입니다. v3는 launch를 더 쓰고 메모리를 크게 줄이는 교환입니다. 실제로 어느 쪽이 빠른지는 이 문서에서 말할 수 없습니다.

### 실행 시간 측정하기

세 실행 파일은 같은 `main.cpp`를 쓰므로 같은 입력과 인자로 돌려 `computation time` 줄을 비교하면 됩니다.

```bash
cd cuda
for v in census_v1 census_v2 census_v3; do
  mkdir -p out_$v
  ./build/$v ../data/sample/right.png ../data/sample/left.png \
      --max-disp 128 --census 7 --aggregate 11 --threshold 1 --no-display \
      --out-dir out_$v | grep "computation time"
done
```

측정값을 읽을 때 알아 둘 점입니다 ([cuda/src/main.cpp:250-256](../cuda/src/main.cpp#L250-L256)).

- 타이머는 `DiffBoxImages()` 호출 하나만 `clock()`으로 감쌉니다. 그 안에는 host 쪽 입력 복사, `cudaMalloc`/`cudaFree`, upload/download, 모든 kernel이 들어 있고, 첫 CUDA 호출의 초기화 비용이 포함될 수 있습니다. 이미지 읽기와 `.png`/`.pfm` 쓰기, 미리보기 창은 포함되지 않습니다.
- `clock()`은 프로세스의 CPU 시간입니다. 각 launch 뒤에 `cudaDeviceSynchronize()`가 있어서 보통 경과 시간과 비슷하게 나오겠지만, 보장되지는 않습니다. kernel별 시간이 필요하면 Nsight Systems(`nsys profile --stats=true ./build/census_v3 ...`)나 `cudaEvent`를 쓰십시오.
- 첫 실행은 캐시와 초기화 때문에 느릴 수 있으므로 여러 번 돌려 반복 사이의 편차를 확인하십시오.
- 세 버전은 소스를 읽은 바로는 같은 disparity를 내야 합니다 (같은 cost, 같은 tie-breaking, 같은 clamp). 각 `out_*/left.pfm`을 NaN 위치까지 포함해 비교하면 fusion과 rolling 구현이 맞는지 확인할 수 있습니다. 다만 이것도 실행해 본 결과가 아니라 코드를 읽은 예상입니다.

## 5. 트레이드오프

- **`VectorShift`가 반복마다 slice 두 개를 복사합니다.** 읽기 2개와 쓰기 2개, 즉 반복당 `16HW` byte(샘플 1216 x 896에서 약 16.6 MiB, 1280 x 720에서는 약 14.1 MiB)의 global memory 요청과 launch 하나가 추가됩니다. 창을 미는 일을 데이터 이동으로 구현했기 때문입니다.
- **엄격한 `<` 동률 처리.** `VectorBox_N_Cost`는 `cost(d-1) < minCosts[1]`일 때만 갱신하므로 동률이면 먼저 나온(작은) disparity가 남습니다. v1/v2의 `min_cost > costs[...]`와 같은 규칙입니다. 후보를 한 단계 늦게 평가해도 후보의 평가 순서는 `0, 1, ..., D-1`로 그대로입니다.
- **disparity마다 launch 세 개.** fusion은 XOR와 Sum까지만 합쳤고, box filter는 여전히 별도 kernel이며 매 launch 뒤에 `cudaDeviceSynchronize()`가 있습니다.
- **box filter가 순진한 O(A^2)입니다.** `VectorBox_N_Cost`의 합산 루프는 v1의 `VectorBox`와 같으므로 비용이 A에 따라 늘어납니다.
- **`minDispValue`는 `ushort`입니다.** `max_disp`가 65535를 넘으면 `disp - 1`을 담을 수 없습니다. 실사용 범위에서는 문제가 되지 않습니다.
- **07장의 확인이 필요한 부분**은 모두 그대로 남아 있습니다 (popcount의 `|`, census 행 누락, sub-pixel 범위 끝, 인자 순서, 경계 clamp).

## 6. 더 해볼 것

독자가 직접 시도해 볼 만한 것들입니다. 효과는 측정해 봐야 알 수 있고, 일부는 07장의 확인이 필요한 부분을 건드리므로 결과 값이 바뀝니다.

- **`__popc`.** 8개 비트 테스트를 `xor_summation += __popc(XOR_byte);`로 바꾸면 코드가 짧아지고 확인이 필요한 부분 1(비트 수가 아닌 byte 수를 세는 것으로 보이는 부분)도 함께 정리됩니다. 더 나아가 census를 4 byte(uint) 단위로 읽으면 load 횟수가 줄어듭니다.
- **2-D block 구성.** 지금은 1차원 `idx = i * W + j`를 `/`, `%`로 분해합니다. `dim3 block(32, 8)`처럼 `threadIdx.x`를 열에 대응시키면 인접 thread가 인접 픽셀을 읽는 형태를 명시적으로 만들 수 있습니다. 또 `VectorCensusXOR_N_Sum`은 인접 thread(인접 픽셀)가 `n` byte 간격으로 census를 읽는 배치이므로, 메모리 배치를 바꿔 보는 것도 방법입니다.
- **separable / integral-image box filter.** A x A 합은 가로 합 후 세로 합으로 나눌 수 있고(픽셀당 O(2A)), integral image를 쓰면 A와 무관한 O(1)이 됩니다 ([03장](03_cost_aggregation.md)). 경계에서 window를 자르는 현재 동작을 그대로 재현하는지 확인해야 합니다.
- **`VectorShift` 대신 포인터 회전.** slice 세 개를 링으로 두고 `(disp + k) % 3`으로 slice 위치만 바꾸면 복사가 필요 없습니다. kernel은 세 slice의 오프셋을 인자로 받게 됩니다.
- **동기화 줄이기.** 같은 stream의 kernel은 순서대로 실행되므로, 오류 확인용 `cudaDeviceSynchronize()`를 루프 밖으로 옮기는 방법을 시험해 볼 수 있습니다.
- **확인이 필요한 부분 검증하기.** 07장의 다섯 항목을 GPU에서 하나씩 확인하고, 필요하면 고친 뒤 [PyTorch 구현](06_pytorch_walkthrough.md)과 출력을 맞춰 보는 것이 가장 좋은 연습입니다.

## 7. 정리

세 장에서 같은 파이프라인을 세 번 구현했습니다.

1. **v1 (07장)**: bit-packed census, grid-stride loop, kernel 세 개, 방향마다 전체 cost volume. 단순하지만 메모리는 `8DHW`에 비례합니다.
2. **v2 (08장)**: XOR와 Sum을 합쳐 중간 buffer와 launch를 줄이고 volume을 하나로 재사용합니다. 메모리는 `4DHW`입니다.
3. **v3 (09장)**: WTA와 sub-pixel에 필요한 것은 최솟값 주변의 세 cost뿐이라는 점을 이용해 3-slice 창과 최솟값 기록으로 대체합니다. 메모리는 D와 무관한 `2nHW + 44HW`입니다.

핵심은 알고리즘이 요구하는 정보만 남기면 메모리 구조를 바꿀 수 있다는 점입니다. 속도는 이 저장소에서 아직 측정하지 않았으므로, 위의 측정 방법으로 직접 확인해 보시기 바랍니다. 전체 구성은 [README](../README.md)로 돌아가 확인하십시오.
