# 08. Kernel fusion: XOR와 Sum 합치기

[07장](07_cuda_baseline.md)의 v1은 disparity마다 `VectorCensusXOR`, `VectorCensusSum`, `VectorBox` 세 kernel을 실행했습니다. 이 장의 v2([cuda/src/v2_fused.cu](../cuda/src/v2_fused.cu))는 앞의 두 kernel을 하나로 합칩니다. 빌드와 실행 방법은 07장과 같고 실행 파일만 `census_v2`입니다. 07장과 마찬가지로 이 코드는 아직 컴파일/실행해 보지 않았으므로 실행 시간은 다루지 않고, 메모리와 launch 횟수처럼 소스에서 셀 수 있는 양만 비교합니다.

## 1. v1에서 v2로: 무엇이 바뀌는가

바뀐 곳은 세 군데입니다.

1. `VectorCensusXOR` + `VectorCensusSum` -> `VectorCensusXOR_N_Sum` (kernel 하나)
2. XOR buffer(`n*H*W` byte)가 사라집니다.
3. cost volume을 left pass와 right pass가 하나로 공유합니다.

나머지 kernel(`VectorCensusTransform`, `VectorBox`, `GetDispFloat`, `GetDispInterp`, `ThresholdDisp`, `GetDispUchar`)은 v1과 같습니다 ([cuda/src/v2_fused.cu:1-9](../cuda/src/v2_fused.cu#L1-L9) 주석).

### kernel: 두 개가 하나로

v1의 두 kernel은 중간 결과 `outputXOR`를 global memory로 주고받았습니다. XOR kernel은 census byte마다 한 thread가 byte 하나를 XOR해서 기록하고 ([cuda/src/v1_baseline.cu:97](../cuda/src/v1_baseline.cu#L97)),

[cuda/src/v1_baseline.cu:97](../cuda/src/v1_baseline.cu#L97)

```cuda
		outputXOR[idx] = leftCensus[idx] ^ rightCensus[i * (img_w * census_size_divide_by_8) + diff * census_size_divide_by_8 + c_i * census_filter_size_divide_by_8 + without_ij_c_i];
```

Sum kernel은 픽셀마다 한 thread가 그 byte들을 다시 읽어 합칩니다 ([cuda/src/v1_baseline.cu:121-122](../cuda/src/v1_baseline.cu#L121-L122)).

[cuda/src/v1_baseline.cu:121-122](../cuda/src/v1_baseline.cu#L121-L122)

```cuda
			uchar XOR_byte = outputXOR[i * (img_w * census_size_divide_by_8) + j * census_size_divide_by_8 + c_i * census_filter_size_divide_by_8 + without_ij_c_i];
			xor_summation += (XOR_byte & 0x01) | ((XOR_byte >> 1) & 0x01) | ((XOR_byte >> 2) & 0x01) | ((XOR_byte >> 3) & 0x01) | ((XOR_byte >> 4) & 0x01) | ((XOR_byte >> 5) & 0x01) | ((XOR_byte >> 6) & 0x01) | ((XOR_byte >> 7) & 0x01);
```

v2에서는 같은 thread가 두 census 배열에서 byte를 직접 읽어 XOR하고 바로 누적합니다 ([cuda/src/v2_fused.cu:86-108](../cuda/src/v2_fused.cu#L86-L108)).

[cuda/src/v2_fused.cu:86-108](../cuda/src/v2_fused.cu#L86-L108)

```cuda
__global__ void VectorCensusXOR_N_Sum(ushort* Census_sum, uchar* leftCensus, uchar* rightCensus, const int census_size, const int census_filter_size, const int img_h, const int img_w, int disp)
{
	const int census_size_divide_by_8 = (int)ceil(census_size / 8.0f);
	const int census_filter_size_divide_by_8 = (int)ceil(census_filter_size / 8.0f);
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		ushort xor_summation = 0;

		int i = idx / img_w;
		int j = idx % img_w;

		for (int without_ij = 0; without_ij < census_size_divide_by_8; without_ij++) {
			int c_i = without_ij / census_filter_size_divide_by_8;
			int without_ij_c_i = without_ij % census_filter_size_divide_by_8;
			int diff = min(max(j + disp, 0), img_w - 1);

            uchar XOR_byte = leftCensus[i * (img_w * census_size_divide_by_8) + j * census_size_divide_by_8 + c_i * census_filter_size_divide_by_8 + without_ij_c_i]
				^ rightCensus[i * (img_w * census_size_divide_by_8) + diff * census_size_divide_by_8 + c_i * census_filter_size_divide_by_8 + without_ij_c_i];
			xor_summation += (XOR_byte & 0x01) | ((XOR_byte >> 1) & 0x01) | ((XOR_byte >> 2) & 0x01) | ((XOR_byte >> 3) & 0x01) | ((XOR_byte >> 4) & 0x01) | ((XOR_byte >> 5) & 0x01) | ((XOR_byte >> 6) & 0x01) | ((XOR_byte >> 7) & 0x01);

		}
		Census_sum[idx] = xor_summation;
	}
}
```

새로 생긴 부분은 `diff = min(max(j + disp, 0), img_w - 1)`을 이 kernel 안에서 계산하는 것(99행)과, 두 census 읽기를 `^`로 잇는 두 줄(101-102행)입니다. 인덱스 산술은 v1의 XOR와 Sum 두 kernel에서 쓰던 것을 그대로 가져왔습니다. 07장의 [bit-packed 배치](07_cuda_baseline.md#4-bit-packed-census의-메모리-배치)가 그대로 적용되며, thread 단위도 Sum과 같은 픽셀(`H*W`)입니다. `disp`가 음수여도 되므로 right pass는 인자 순서를 바꾸고 `-disp`를 넘기는 방식이 그대로입니다.

확인이 필요한 부분 1(`|`로 합치는 popcount)도 이 kernel에 [103행](../cuda/src/v2_fused.cu#L103)으로 그대로 따라왔습니다. 합치기만 했을 뿐 계산은 바꾸지 않았으므로 v1과 v2는 같은 cost를 냅니다.

### host 루프: launch 세 개가 두 개로

v1의 left pass 루프입니다 ([cuda/src/v1_baseline.cu:324-336](../cuda/src/v1_baseline.cu#L324-L336)).

[cuda/src/v1_baseline.cu:324-336](../cuda/src/v1_baseline.cu#L324-L336)

```cuda
	for (int disp = 0; disp < max_disp; disp++) {
		VectorCensusXOR <<<kNumBlocks, kNumThreads>>> (d_leftXOR_mem, d_leftCensus_mem, d_rightCensus_mem, census_size, census_filter_size, img_height, img_width, disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorCensusSum <<<kNumBlocks, kNumThreads>>> (d_leftCensusSum_mem, d_leftXOR_mem, census_size, census_filter_size, img_height, img_width);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorBox <<<kNumBlocks, kNumThreads>>> (d_leftCosts_mem, d_leftCensusSum_mem, aggreate_filter_size, img_height, img_width, disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
	}
```

v2의 같은 부분입니다 ([cuda/src/v2_fused.cu:291-299](../cuda/src/v2_fused.cu#L291-L299)).

[cuda/src/v2_fused.cu:291-299](../cuda/src/v2_fused.cu#L291-L299)

```cuda
	for (int disp = 0; disp < max_disp; disp++) {
		VectorCensusXOR_N_Sum <<<kNumBlocks, kNumThreads>>> (d_CensusSum_mem, d_leftCensus_mem, d_rightCensus_mem, census_size, census_filter_size, img_height, img_width, disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorBox <<<kNumBlocks, kNumThreads>>> (d_Costs_mem, d_CensusSum_mem, aggreate_filter_size, img_height, img_width, disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
	}
```

### buffer: XOR가 사라지고 volume이 하나로

v1은 방향마다 XOR, CensusSum, Costs를 새로 할당하고 해제했습니다. v2는 CensusSum과 Costs를 한 번만 할당합니다 ([cuda/src/v2_fused.cu:286-289](../cuda/src/v2_fused.cu#L286-L289)).

[cuda/src/v2_fused.cu:286-289](../cuda/src/v2_fused.cu#L286-L289)

```cuda
	checkCudaErrors(cudaMalloc(&d_CensusSum_mem, img_height * img_width * sizeof(ushort)));

	// Costs cudaMalloc
	checkCudaErrors(cudaMalloc(&d_Costs_mem, max_disp * img_height * img_width * sizeof(uint)));
```

volume 공유가 가능한 이유는 순서를 바꿨기 때문입니다. v1은 두 방향의 volume이 모두 채워진 뒤에 `GetDispFloat`를 두 번 불렀습니다 ([v1:372-373](../cuda/src/v1_baseline.cu#L372-L373)). v2는 left pass가 끝나자마자 left disparity를 뽑고, 그 다음에야 같은 volume을 right pass가 덮어씁니다 ([cuda/src/v2_fused.cu:301-316](../cuda/src/v2_fused.cu#L301-L316)).

[cuda/src/v2_fused.cu:301-316](../cuda/src/v2_fused.cu#L301-L316)

```cuda
	// Disp cudaMalloc
	checkCudaErrors(cudaMalloc(&d_leftDispFloat_mem, img_height * img_width * sizeof(float)));

	GetDispFloat <<<kNumBlocks, kNumThreads>>> (d_leftDispFloat_mem, d_Costs_mem, sub_pixel, img_height, img_width, max_disp);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	for (int disp = 0; disp < max_disp; disp++) {
		VectorCensusXOR_N_Sum <<<kNumBlocks, kNumThreads>>> (d_CensusSum_mem, d_rightCensus_mem, d_leftCensus_mem, census_size, census_filter_size, img_height, img_width, -disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorBox <<<kNumBlocks, kNumThreads>>> (d_Costs_mem, d_CensusSum_mem, aggreate_filter_size, img_height, img_width, disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
	}
```

left pass는 slice `0..D-1`을 모두 쓰고, right pass도 `0..D-1`을 모두 다시 쓰므로 덮어쓰기에 문제가 없습니다. 이후 `GetDispFloat`를 한 번 더 호출하고 ([323행](../cuda/src/v2_fused.cu#L323)) volume을 해제합니다 ([326행](../cuda/src/v2_fused.cu#L326)).

## 2. 왜 fusion이 도움이 되는가

**kernel launch 횟수.** disparity마다 launch가 3개에서 2개로 줄어듭니다. 각 launch는 host 쪽 호출 비용과 `cudaDeviceSynchronize()`를 동반하므로, 이 횟수가 곧 host가 GPU를 기다리는 횟수입니다.

**중간 결과의 global memory 왕복.** kernel 사이에서 값을 넘기는 유일한 방법은 global memory입니다. v1에서는 XOR 결과 `n*H*W` byte를 쓰고, 다음 kernel이 그것을 다시 읽었습니다. fusion하면 XOR 결과가 thread의 레지스터(`XOR_byte`)에만 머뭅니다. disparity 한 번에 kernel이 global memory에 요청하는 byte 수를 세면 다음과 같습니다 (캐시 효과나 실제 DRAM traffic은 측정하지 않았고, 소스 코드의 load/store만 센 값입니다).

| 단계 | v1 | v2 |
|---|---|---|
| XOR | 읽기 2nHW (left, right) + 쓰기 nHW | (Sum과 합쳐짐) |
| Sum | 읽기 nHW + 쓰기 2HW | 읽기 2nHW + 쓰기 2HW |
| 합계 | **4nHW + 2HW** | **2nHW + 2HW** |

K = 11(n = 16)이면 v1은 66HW, v2는 34HW byte이고, 1280 x 720(HW = 921,600)에서 약 58.0 MiB 대 29.9 MiB입니다. K = 7(n = 7)이면 30HW 대 16HW이고, 샘플 1216 x 896(HW = 1,089,536)에서 약 31.2 MiB 대 16.6 MiB입니다. 줄어든 것은 정확히 XOR 중간 buffer의 쓰기와 읽기(2nHW) 입니다.

**메모리.** XOR buffer `nHW`가 없어지고, volume이 방향별로 둘이던 것이 하나가 됩니다.

## 3. 메모리와 launch 정리: v1 대 v2

v2의 device buffer는 다음과 같습니다 ([cuda/src/v2_fused.cu:1-9](../cuda/src/v2_fused.cu#L1-L9) 주석과 host 코드 [286-326행](../cuda/src/v2_fused.cu#L286-L326) 기준).

| buffer | 크기 (byte) |
|---|---|
| census x2 | 2nHW |
| CensusSum (ushort) | 2HW |
| Costs (uint, 하나) | 4DHW |
| left float disparity (right pass 중에 이미 존재) | 4HW |

peak는 right pass 중이며 다음과 같습니다.

> peak(v2) = 2nHW + 2HW + 4DHW + 4HW = **2nHW + 6HW + 4DHW**

소스 header 주석은 left float disparity map(`4HW`, 302행에서 할당되어 right pass 내내 살아 있음)을 뺀 `2nHW + 2HW + 4DHW`라고 적었습니다 ("about"으로 표현된 어림식). 이 문서는 host 코드의 할당 순서를 따라 `4HW`를 포함하며, 그 차이는 1280 x 720에서 3.5 MiB입니다 (K = 11, D = 300에서 header 식은 1084.6 MiB, 이 식은 1088.1 MiB).

launch 횟수는 census 2 + 방향별 `2D`(XOR_N_Sum, Box) x 2 + 나머지 8 = **4D + 10**입니다.

| | v1 | v2 |
|---|---|---|
| kernel 종류 | 8 | 7 |
| disparity당 launch (방향당) | 3 | 2 |
| 전체 launch | 6D + 10 | 4D + 10 |
| XOR buffer | nHW (방향마다 할당) | 없음 |
| cost volume | 4DHW x 2 (공존) | 4DHW x 1 (재사용) |
| peak 메모리 | 3nHW + 2HW + 8DHW | 2nHW + 6HW + 4DHW |
| 1280x720, K = 11, D = 300 | 2153.3 MiB | 1088.1 MiB |
| 샘플 1216x896, K = 7, D = 128 | 1087.9 MiB | 552.8 MiB |
| D = 300 launch 수 (해상도 무관) | 1810 | 1210 |
| D = 128 launch 수 (해상도 무관) | 778 | 522 |
| 실행 시간 | 측정 필요 | 측정 필요 |

계산 근거: 1280 x 720은 HW = 921,600, 샘플 1216 x 896은 HW = 1,089,536이고, MiB = 2^20 byte. K = 11이면 n = 16, K = 7이면 n = 7입니다. D가 큰 설정에서는 volume 항(`8DHW` 대 `4DHW`)이 지배적이라 메모리가 거의 절반이 됩니다.

## 4. 여전히 비싼 것

v2에서도 남은 가장 큰 비용은 **cost volume `4DHW` byte**입니다. 1280 x 720, K = 11, D = 300에서 1054.7 MiB, 샘플 설정(1216 x 896, D = 128)에서도 532.0 MiB로, 전체 peak의 대부분입니다. 이 volume은 다음과 같이 쓰입니다.

- `VectorBox`가 disparity마다 slice 하나를 씁니다 (`costs[disp * img_h * img_w + idx]`, [v2:137](../cuda/src/v2_fused.cu#L137)).
- `GetDispFloat`가 모든 slice를 한 번씩 읽어 최솟값과 그 이웃 세 값만 취합니다 ([v2:155-160](../cuda/src/v2_fused.cu#L155-L160)).

즉 volume은 한 번 쓰이고 한 번 읽힐 뿐이며, 읽은 결과는 픽셀당 값 몇 개(최솟값 위치와 이웃 비용 둘)로 줄어듭니다. WTA와 sub-pixel 계산에 필요한 것이 정말 그 세 비용뿐이라면 volume 전체를 보관할 이유가 없습니다. [09. Rolling buffer: max_disp와 무관한 메모리](09_cuda_rolling_buffer.md)에서 그 아이디어를 구현합니다.
