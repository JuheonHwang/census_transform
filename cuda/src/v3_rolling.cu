// v3_rolling.cu -- stage 3 (final): fused kernel + rolling 3-slice cost buffers.
// No cost volume is stored: VectorBox_N_Cost keeps the running minimum (and the two neighbouring
// costs needed for the sub-pixel fit) per pixel, so memory does not depend on D. Costs D + 1
// iterations per direction, each = VectorShift + VectorCensusXOR_N_Sum + VectorBox_N_Cost.
// Kernels: VectorCensusTransform, VectorCensusXOR_N_Sum, VectorBox_N_Cost, VectorShift,
//          GetDispFloat, GetDispInterp, ThresholdDisp, GetDispUchar.
// Device buffers (bytes): images 2*H*W (freed after census); census 2*n*H*W; CensusSum 2*H*W;
//   Cost_a, Cost_b 2 * 12*H*W (3 uint slices each); minCosts 12*H*W; minDispValue 2*H*W;
//   float disparity 2 * 4*H*W (left allocated before the right pass); interpolation 2 * 4*H*W;
//   uchar output 2 * H*W. Total is about 2*n*H*W + 40*H*W + ... independent of D.
// Also returns the float disparity maps (for PFM output) in addition to the uchar visualisation.
//
// Notation used in all comments of this file
//   H, W        image height / width            D = max_disp
//   K           census_filter_size (census window is K x K)
//   A           aggreate_filter_size (box-filter window is A x A)
//   n           = ceil(K*K / 8) = bytes of census code per pixel
//   B           = ceil(K / 8)   = bytes per census window row
//   idx         linear pixel index  idx = i * W + j   (i = row, j = column)
//
// Bit-packed census buffer (uchar, H * W * n bytes):
//   census[(i * W + j) * n + b],  b = 0 .. n-1
//   Byte b of pixel (i, j) encodes window row c_i = b / B, window columns
//   (b % B) * 8 .. (b % B) * 8 + 7. Bit = (centre pixel < neighbour), MSB first.
//   The last byte of a window row holds only K % 8 bits (right aligned).
//
// Disparity sign convention: the "left" image is the reference; its pixel
// (i, j) is compared with the "right" image pixel (i, j + disp). The right
// pass uses j - disp. Border coordinates are clamped (replicated).

#include "census.cuh"
#include <stdio.h>
#include <limits.h>
#define POW(X) ((X)*(X))

// VectorCensusTransform: bit-packed census transform of one image.
//   Output : censusTransform[(i * W + j) * n + b]  (layout described at the top of this file)
//   Threads: one thread per OUTPUT BYTE, grid-stride loop over idx in [0, n * H * W).
//   Index decomposition of idx:
//     i = idx / (W * n)                row
//     j = (idx % (W * n)) / n          column
//     b = idx % n                      byte within the pixel's code
//     c_i = b / B                      census window row this byte covers
//     c_j = (b % B) * 8 + bit          census window column of each bit
//   Bit value: image(i, j) < image(i + c_i - K/2, j + c_j - K/2), coordinates clamped to the image.
//   NOTE (existing quirk, not changed): n = ceil(K*K/8) is smaller than K * B unless K is a
//   multiple of 8, so window rows c_i > (n - 1) / B are never encoded (K = 11: rows 0..7 of 0..10).
__global__ void VectorCensusTransform(uchar* censusTransform, uchar* image, const int census_size, const int census_filter_size, const int img_h, const int img_w)
{
	const int census_half = census_filter_size / 2;
	const int census_size_divide_by_8 = (int)ceil(census_size / 8.0f);
	const int census_filter_size_divide_by_8 = (int)ceil(census_filter_size / 8.0f);
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < census_size_divide_by_8 * img_h * img_w; idx += blockDim.x * gridDim.x) {
		int i = idx / (img_w * census_size_divide_by_8); int without_i = idx % (img_w * census_size_divide_by_8);
		int j = without_i / census_size_divide_by_8; int without_ij = without_i % census_size_divide_by_8;

		int c_i = without_ij / census_filter_size_divide_by_8;
		int without_ij_c_i = without_ij % census_filter_size_divide_by_8;

		uchar census = 0;
		uchar image_center = image[img_w * i + j];

		for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
			int c_j = without_ij_c_i * 8 + bit_idx;

			if (c_j >= census_filter_size)
				break;

			int c_i_2_image_i = min(max(c_i - census_half + i, 0), img_h - 1);
			int c_j_2_image_j = min(max(c_j - census_half + j, 0), img_w - 1);

			census <<= 1;
			census |= (image_center < image[img_w * c_i_2_image_i + c_j_2_image_j]);
		}
		censusTransform[idx] = census;
	}
}

// VectorCensusXOR_N_Sum (v2, v3): fused XOR + sum, i.e. VectorCensusXOR + VectorCensusSum in one kernel.
//   Output : Census_sum[idx] (ushort, H * W), pixel-wise cost for ONE disparity `disp`.
//   Threads: one thread per PIXEL (grid-stride loop over idx in [0, H * W)).
//   Each thread walks the n bytes of its pixel: byte b of the left code at (i, j) is XORed with
//   byte b of the right code at (i, clamp(j + disp, 0, W-1)); no n * H * W XOR buffer is needed.
//   disp may be negative (right pass: arguments swapped and disp negated).
//   NOTE (existing quirk, not changed): the per-byte "popcount" is written as an OR of the 8 bits
//   ((x & 1) | ((x >> 1) & 1) | ...), which yields 0/1 per byte, i.e. it counts non-zero XOR bytes
//   rather than differing bits. A true Hamming weight would use '+' (or __popc).
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


// VectorBox_N_Cost (v3): box filter + running minimum over disparity (the rolling-buffer scheme).
//   Instead of storing D cost slices, v3 keeps three H * W slices per buffer:
//     cost_dummy[0 * H * W + idx] = cost(d - 2)   cost_dummy[1 * H * W + idx] = cost(d - 1)
//     cost_dummy[2 * H * W + idx] = cost(d)       (d = the `disp` argument)
//   and a "best so far" record for each pixel:
//     minCosts[0..2][idx] = (cost(d*-1), cost(d*), cost(d*+1)) at the best disparity d* so far (uint)
//     minDispValue[idx]   = d*                                                         (ushort)
//   Host loop (see DiffBoxImages): two buffers a / b are ping-ponged. For each disp >= 1 the host
//   calls VectorShift(curr, prev) (curr[0] = prev[1], curr[1] = prev[2]), then
//   VectorCensusXOR_N_Sum, then this kernel with cost_dummy = curr.
//   This kernel:
//     1. aggregates Census_sum over the A x A window (same as VectorBox), giving cost(disp);
//     2. disp == 0: fills slots 1 and 2 with it (so cost(-1) is clamped to cost(0) after the shift);
//        disp  > 0: stores it in slot 2, then tests the MIDDLE slot cost(disp - 1) against
//        minCosts[1]; if strictly smaller, records minDispValue = disp - 1 and copies all three
//        slots to minCosts. The candidate is therefore evaluated one step late, when its
//        "next" neighbour is available for the sub-pixel fit.
//   The host runs disp = 0 .. D, so the last candidate D-1 gets its "next" slice from the clamped
//   XOR at min(disp, D-1) (a duplicate of cost(D-1), matching the clamping in v1/v2).
//   minCosts must be initialised to 0xFFFFFFFF (cudaMemset with 0xFF bytes) before each pass.
//   Threads: one thread per PIXEL; each thread only touches its own idx, so there are no races.
__global__ void VectorBox_N_Cost(uint* minCosts, ushort* minDispValue, uint* cost_dummy, ushort* Census_sum, const int aggreate_filter_size, const int img_h, const int img_w, int disp)
{
	const int aggreate_half = (aggreate_filter_size - 1) / 2;

	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		uint cost_summation = 0;

		int i = idx / img_w;
		int j = idx % img_w;

		const int height_begin = max(i - aggreate_half, 0);
		const int height_end = min(i + aggreate_half + 1, img_h);
		const int width_begin = max(j - aggreate_half, 0);
		const int width_end = min(j + aggreate_half + 1, img_w);

		for (int ii = height_begin; ii < height_end; ii++) {
			for (int jj = width_begin; jj < width_end; jj++) {
				cost_summation += Census_sum[ii * img_w + jj];
			}
		}

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
	}
}

// VectorShift (v3): slide the 3-slice cost window by one disparity.
//   cost_curr[0] = cost_prev[1]; cost_curr[1] = cost_prev[2]   (slice 2 is written by VectorBox_N_Cost).
//   Threads: one thread per PIXEL; slices are H * W uints each (index s * H * W + idx).
//   The host alternates which of the two buffers is `curr` / `prev` on every disparity.
__global__ void VectorShift(uint* cost_curr, uint* cost_prev, const int img_h, const int img_w)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		cost_curr[0 * img_h * img_w + idx] = cost_prev[1 * img_h * img_w + idx];
		cost_curr[1 * img_h * img_w + idx] = cost_prev[2 * img_h * img_w + idx];
	}
}

// GetDispFloat (v3): turn the recorded best disparity into a float disparity map.
//   Output : disp_map_float[idx].
//   Threads: one thread per PIXEL. No scan over disparity: the winner was already tracked.
//   Sub-pixel: disp = minDispValue + (next - prev) / (2 * (2 * curr - prev - next)) with
//   prev / curr / next = minCosts[0 / 1 / 2][idx]; otherwise disp = minDispValue.
//   Caveat (not changed): the denominator is 0 on flat cost curves (NaN/inf), and at d* = 0 or
//   D-1 the duplicated neighbour gives an offset of -0.5 or +0.5.
__global__ void GetDispFloat(float* disp_map_float, uint* minCosts, ushort* minDispValue, const bool sub_pixel, const int img_h, const int img_w, const int max_disp)
{
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
}

// GetDispInterp: sample the OTHER image's disparity map at the position this pixel maps to
//   (left-right consistency, step 1). Output: interp_disp_map[idx] (float).
//   Threads: one thread per PIXEL.
//   x' = clamp(j + right2left * opt_disp_map[idx], 0, W - 1); result = linear interpolation of
//   target_disp_map(i, x') between columns floor(x') and floor(x') + 1.
//   `right2left` is really a sign (+1 / -1): +1 maps a left pixel to the right image
//   (x + d), -1 maps a right pixel to the left image (x - d), matching the search direction above.
__global__ void GetDispInterp(float* interp_disp_map, float* target_disp_map, float* opt_disp_map, const int right2left, const int img_h, const int img_w)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		int i = idx / img_w;
		int j = idx % img_w;

		float target_idx = min(max((float)j + (float)right2left * opt_disp_map[idx], 0.0f), (float)img_w - 1);
		int target_idx1 = (int)target_idx;
		int target_idx2 = min((int)target_idx + 1, img_w - 1);
		float alpha = target_idx - (int)target_idx;

		interp_disp_map[idx] = alpha * target_disp_map[i * img_w + target_idx2] + (1 - alpha) * target_disp_map[i * img_w + target_idx1];
	}
}

// ThresholdDisp: left-right consistency check (step 2). In place on disp_map.
//   Threads: one thread per PIXEL. If |interp_disp_map - disp_map| > threshold (disparity units)
//   the pixel is marked invalid by setting its disparity to 0.
__global__ void ThresholdDisp(float* disp_map, float* interp_disp_map, const int threshold, const int img_h, const int img_w)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		if (abs(interp_disp_map[idx] - disp_map[idx]) > threshold)
		{
			disp_map[idx] = 0;
		}
	}
}

// GetDispUchar: visualisation. disp_map_uchar = (uchar) min(256 / D * disp, 255).
//   Threads: one thread per PIXEL. Negative or NaN disparities are not guarded (the cast is undefined).
__global__ void GetDispUchar(uchar* disp_map_uchar, float* disp_map_float, const int img_h, const int img_w, const int max_disp)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		disp_map_uchar[idx] = (uchar)min((256.0f / (float)max_disp) * disp_map_float[idx], 255.0f);
	}
}

__host__ void DiffBoxImages(float* pLeftOutPfm, float* pRightOutPfm, uchar* pLeftOutImg, uchar* pRightOutImg, uchar* pimageLeft, uchar* pimageRight,
	const bool sub_pixel, const int threshold, const int census_filter_size, const int aggreate_filter_size,
	const int img_height, const int img_width, const int max_disp)
{
	uchar* imageLeft_mem;
	uchar* imageRight_mem;

	uchar* d_imageLeft_mem;
	uchar* d_imageRight_mem;


	uchar* d_leftCensus_mem;
	uchar* d_rightCensus_mem;

	ushort* d_CensusSum_mem;

	uint* d_Cost_a_mem;
	uint* d_Cost_b_mem;

	uint* d_minCosts_mem;
	ushort* d_minDispValue_mem;

	float* d_leftDispFloat_mem;
	float* d_rightDispFloat_mem;

	float* d_left2rightDisp_mem;
	float* d_right2leftDisp_mem;

	uchar* d_leftDispUchar_mem;
	uchar* d_rightDispUchar_mem;

	const int census_size = POW(census_filter_size);


	//int i = 1;
	//for (; i < census_size; i *= 8);
	int n = static_cast<int>(ceil(census_size / 8.0f));

	// Image cudaMalloc
	imageLeft_mem = (uchar*)malloc(img_height * img_width * sizeof(uchar));
	memcpy(imageLeft_mem, pimageLeft, sizeof(uchar) * img_height * img_width);

	imageRight_mem = (uchar*)malloc(img_height * img_width * sizeof(uchar));
	memcpy(imageRight_mem, pimageRight, sizeof(uchar) * img_height * img_width);

	checkCudaErrors(cudaMalloc(&d_imageLeft_mem, img_height * img_width * sizeof(uchar)));
	checkCudaErrors(cudaMemcpy(d_imageLeft_mem, imageLeft_mem, img_height * img_width * sizeof(uchar), cudaMemcpyHostToDevice));
	free(imageLeft_mem);

	checkCudaErrors(cudaMalloc(&d_imageRight_mem, img_height * img_width * sizeof(uchar)));
	checkCudaErrors(cudaMemcpy(d_imageRight_mem, imageRight_mem, img_height * img_width * sizeof(uchar), cudaMemcpyHostToDevice));
	free(imageRight_mem);


	// Census cudaMalloc
	checkCudaErrors(cudaMalloc(&d_leftCensus_mem, n * img_height * img_width * sizeof(uchar)));
	checkCudaErrors(cudaMemset(d_leftCensus_mem, 0, n * img_height * img_width * sizeof(uchar)));

	checkCudaErrors(cudaMalloc(&d_rightCensus_mem, n * img_height * img_width * sizeof(uchar)));
	checkCudaErrors(cudaMemset(d_rightCensus_mem, 0, n * img_height * img_width * sizeof(uchar)));

	VectorCensusTransform <<<kNumBlocks, kNumThreads>>> (d_leftCensus_mem, d_imageLeft_mem, census_size, census_filter_size, img_height, img_width);
	VectorCensusTransform <<<kNumBlocks, kNumThreads>>> (d_rightCensus_mem, d_imageRight_mem, census_size, census_filter_size, img_height, img_width);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
	checkCudaErrors(cudaFree(d_imageLeft_mem)); checkCudaErrors(cudaFree(d_imageRight_mem));

	// CensusSum cudaMalloc
	checkCudaErrors(cudaMalloc(&d_CensusSum_mem, img_height * img_width * sizeof(ushort)));

	// Costs cudaMalloc
	checkCudaErrors(cudaMalloc(&d_Cost_a_mem, 3 * img_height * img_width * sizeof(uint)));
	checkCudaErrors(cudaMalloc(&d_Cost_b_mem, 3 * img_height * img_width * sizeof(uint)));

	checkCudaErrors(cudaMalloc(&d_minCosts_mem, 3 * img_height * img_width * sizeof(uint)));
	// cudaMemset fills BYTES: UINT_MAX is truncated to 0xFF, so every uint becomes 0xFFFFFFFF (= UINT_MAX)
	checkCudaErrors(cudaMemset(d_minCosts_mem, UINT_MAX, 3 * img_height * img_width * sizeof(uint)));

	checkCudaErrors(cudaMalloc(&d_minDispValue_mem, img_height * img_width * sizeof(ushort)));

	// left
	// initialize
	VectorCensusXOR_N_Sum <<<kNumBlocks, kNumThreads>>> (d_CensusSum_mem, d_leftCensus_mem, d_rightCensus_mem, census_size, census_filter_size, img_height, img_width, 0);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	VectorBox_N_Cost <<<kNumBlocks, kNumThreads>>> (d_minCosts_mem, d_minDispValue_mem, d_Cost_a_mem, d_CensusSum_mem, aggreate_filter_size, img_height, img_width, 0);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	
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

	// Disp cudaMalloc
	checkCudaErrors(cudaMalloc(&d_leftDispFloat_mem, img_height * img_width * sizeof(float)));

	GetDispFloat <<<kNumBlocks, kNumThreads>>> (d_leftDispFloat_mem, d_minCosts_mem, d_minDispValue_mem, sub_pixel, img_height, img_width, max_disp);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	// right
	// initialize
	// cudaMemset fills BYTES: UINT_MAX is truncated to 0xFF, so every uint becomes 0xFFFFFFFF (= UINT_MAX)
	checkCudaErrors(cudaMemset(d_minCosts_mem, UINT_MAX, 3 * img_height * img_width * sizeof(uint)));

	VectorCensusXOR_N_Sum <<<kNumBlocks, kNumThreads>>> (d_CensusSum_mem, d_rightCensus_mem, d_leftCensus_mem, census_size, census_filter_size, img_height, img_width, 0);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	VectorBox_N_Cost <<<kNumBlocks, kNumThreads>>> (d_minCosts_mem, d_minDispValue_mem, d_Cost_a_mem, d_CensusSum_mem, aggreate_filter_size, img_height, img_width, 0);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	for (int disp = 1; disp < (max_disp + 1); disp++) {
		// shift
		VectorShift <<<kNumBlocks, kNumThreads>>> (((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), ((disp % 2) ? d_Cost_a_mem : d_Cost_b_mem), img_height, img_width);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorCensusXOR_N_Sum <<<kNumBlocks, kNumThreads>>> (d_CensusSum_mem, d_rightCensus_mem, d_leftCensus_mem, census_size, census_filter_size, img_height, img_width, -min(disp, max_disp - 1));
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorBox_N_Cost <<<kNumBlocks, kNumThreads>>> (d_minCosts_mem, d_minDispValue_mem, ((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), d_CensusSum_mem, aggreate_filter_size, img_height, img_width, disp);
		cudaDeviceSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
	}
	checkCudaErrors(cudaFree(d_leftCensus_mem)); checkCudaErrors(cudaFree(d_rightCensus_mem));
	checkCudaErrors(cudaFree(d_CensusSum_mem));
	checkCudaErrors(cudaFree(d_Cost_a_mem)); checkCudaErrors(cudaFree(d_Cost_b_mem));

	// Disp cudaMalloc
	checkCudaErrors(cudaMalloc(&d_rightDispFloat_mem, img_height * img_width * sizeof(float)));

	GetDispFloat <<<kNumBlocks, kNumThreads>>> (d_rightDispFloat_mem, d_minCosts_mem, d_minDispValue_mem, sub_pixel, img_height, img_width, max_disp);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
	checkCudaErrors(cudaFree(d_minCosts_mem));
	checkCudaErrors(cudaFree(d_minDispValue_mem));

	cudaMalloc(&d_left2rightDisp_mem, img_height * img_width * sizeof(float));
	cudaMalloc(&d_right2leftDisp_mem, img_height * img_width * sizeof(float));

	GetDispInterp <<<kNumBlocks, kNumThreads>>> (d_right2leftDisp_mem, d_rightDispFloat_mem, d_leftDispFloat_mem, 1, img_height, img_width);
	GetDispInterp <<<kNumBlocks, kNumThreads>>> (d_left2rightDisp_mem, d_leftDispFloat_mem, d_rightDispFloat_mem, -1, img_height, img_width);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	ThresholdDisp <<<kNumBlocks, kNumThreads>>> (d_leftDispFloat_mem, d_right2leftDisp_mem, threshold, img_height, img_width);
	ThresholdDisp <<<kNumBlocks, kNumThreads>>> (d_rightDispFloat_mem, d_left2rightDisp_mem, threshold, img_height, img_width);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
	checkCudaErrors(cudaFree(d_left2rightDisp_mem)); checkCudaErrors(cudaFree(d_right2leftDisp_mem));


	checkCudaErrors(cudaMalloc(&d_leftDispUchar_mem, img_height * img_width * sizeof(uchar)));
	checkCudaErrors(cudaMalloc(&d_rightDispUchar_mem, img_height * img_width * sizeof(uchar)));

	GetDispUchar <<<kNumBlocks, kNumThreads>>> (d_leftDispUchar_mem, d_leftDispFloat_mem, img_height, img_width, max_disp);
	GetDispUchar <<<kNumBlocks, kNumThreads>>> (d_rightDispUchar_mem, d_rightDispFloat_mem, img_height, img_width, max_disp);
	cudaDeviceSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
	cudaMemcpy(pLeftOutPfm, d_leftDispFloat_mem, img_height * img_width * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaFree(d_leftDispFloat_mem));
	cudaMemcpy(pRightOutPfm, d_rightDispFloat_mem, img_height * img_width * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaFree(d_rightDispFloat_mem));


	cudaMemcpy(pLeftOutImg, d_leftDispUchar_mem, img_height * img_width * sizeof(uchar), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaFree(d_leftDispUchar_mem));

	cudaMemcpy(pRightOutImg, d_rightDispUchar_mem, img_height * img_width * sizeof(uchar), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaFree(d_rightDispUchar_mem));
}

__host__ void checkCUDA(const int lineNumber, cudaError_t status) {
	if (status != cudaSuccess) {
		fprintf(stderr, "CUDA failure at LINE %d : %s - %s\n", lineNumber, cudaGetErrorName(status), cudaGetErrorString(status));
		FatalError(lineNumber);
	}
}

__host__ void FatalError(const int lineNumber) {
	fprintf(stderr, "FatalError");
	if (lineNumber != 0) fprintf(stderr, " at LINE %d", lineNumber);
	fprintf(stderr, ". Program Terminated.\n");
	cudaDeviceReset();
	exit(EXIT_FAILURE);
}