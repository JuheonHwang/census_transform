#include "census.cuh"
#include <stdio.h>
#define POW(X) ((X)*(X))

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

__global__ void VectorShift(uint* cost_curr, uint* cost_prev, const int img_h, const int img_w)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		cost_curr[0 * img_h * img_w + idx] = cost_prev[1 * img_h * img_w + idx];
		cost_curr[1 * img_h * img_w + idx] = cost_prev[2 * img_h * img_w + idx];
	}
}

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

__global__ void ThresholdDisp(float* disp_map, float* interp_disp_map, const int threshold, const int img_h, const int img_w)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < img_h * img_w; idx += blockDim.x * gridDim.x) {
		if (abs(interp_disp_map[idx] - disp_map[idx]) > threshold)
		{
			disp_map[idx] = 0;
		}
	}
}

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


	VectorCensusTransform << <128, 128 >> > (d_leftCensus_mem, d_imageLeft_mem, census_size, census_filter_size, img_height, img_width);
	VectorCensusTransform << <128, 128 >> > (d_rightCensus_mem, d_imageRight_mem, census_size, census_filter_size, img_height, img_width);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
	checkCudaErrors(cudaFree(d_imageLeft_mem)); checkCudaErrors(cudaFree(d_imageRight_mem));

	// CensusSum cudaMalloc
	checkCudaErrors(cudaMalloc(&d_CensusSum_mem, img_height * img_width * sizeof(ushort)));

	// Costs cudaMalloc
	checkCudaErrors(cudaMalloc(&d_Cost_a_mem, 3 * img_height * img_width * sizeof(uint)));
	checkCudaErrors(cudaMalloc(&d_Cost_b_mem, 3 * img_height * img_width * sizeof(uint)));

	checkCudaErrors(cudaMalloc(&d_minCosts_mem, 3 * img_height * img_width * sizeof(uint)));
	checkCudaErrors(cudaMemset(d_minCosts_mem, UINT_MAX, 3 * img_height * img_width * sizeof(uint)));

	checkCudaErrors(cudaMalloc(&d_minDispValue_mem, img_height * img_width * sizeof(ushort)));

	// left
	// initialize
	VectorCensusXOR_N_Sum << <128, 128 >> > (d_CensusSum_mem, d_leftCensus_mem, d_rightCensus_mem, census_size, census_filter_size, img_height, img_width, 0);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	VectorBox_N_Cost << <128, 128 >> > (d_minCosts_mem, d_minDispValue_mem, d_Cost_a_mem, d_CensusSum_mem, aggreate_filter_size, img_height, img_width, 0);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	
	for (int disp = 1; disp < (max_disp + 1); disp++) {
		// shift
		VectorShift << <128, 128 >> > (((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), ((disp % 2) ? d_Cost_a_mem : d_Cost_b_mem), img_height, img_width);
        cudaThreadSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
		
		VectorCensusXOR_N_Sum << <128, 128 >> > (d_CensusSum_mem, d_leftCensus_mem, d_rightCensus_mem, census_size, census_filter_size, img_height, img_width, min(disp, max_disp - 1));
		cudaThreadSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorBox_N_Cost << <128, 128 >> > (d_minCosts_mem, d_minDispValue_mem, ((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), d_CensusSum_mem, aggreate_filter_size, img_height, img_width, disp);
		cudaThreadSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
	}

	// Disp cudaMalloc
	checkCudaErrors(cudaMalloc(&d_leftDispFloat_mem, img_height * img_width * sizeof(float)));

	GetDispFloat << <128, 128 >> > (d_leftDispFloat_mem, d_minCosts_mem, d_minDispValue_mem, sub_pixel, img_height, img_width, max_disp);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	// right
	// initialize
	checkCudaErrors(cudaMemset(d_minCosts_mem, UINT_MAX, 3 * img_height * img_width * sizeof(uint)));

	VectorCensusXOR_N_Sum << <128, 128 >> > (d_CensusSum_mem, d_rightCensus_mem, d_leftCensus_mem, census_size, census_filter_size, img_height, img_width, 0);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	VectorBox_N_Cost << <128, 128 >> > (d_minCosts_mem, d_minDispValue_mem, d_Cost_a_mem, d_CensusSum_mem, aggreate_filter_size, img_height, img_width, 0);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	for (int disp = 1; disp < (max_disp + 1); disp++) {
		// shift
		VectorShift << <128, 128 >> > (((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), ((disp % 2) ? d_Cost_a_mem : d_Cost_b_mem), img_height, img_width);
		cudaThreadSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorCensusXOR_N_Sum << <128, 128 >> > (d_CensusSum_mem, d_rightCensus_mem, d_leftCensus_mem, census_size, census_filter_size, img_height, img_width, -min(disp, max_disp - 1));
		cudaThreadSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());

		VectorBox_N_Cost << <128, 128 >> > (d_minCosts_mem, d_minDispValue_mem, ((disp % 2) ? d_Cost_b_mem : d_Cost_a_mem), d_CensusSum_mem, aggreate_filter_size, img_height, img_width, disp);
		cudaThreadSynchronize();
		checkCUDA(__LINE__, cudaGetLastError());
	}
	checkCudaErrors(cudaFree(d_leftCensus_mem)); checkCudaErrors(cudaFree(d_rightCensus_mem));
	checkCudaErrors(cudaFree(d_CensusSum_mem));

	// Disp cudaMalloc
	checkCudaErrors(cudaMalloc(&d_rightDispFloat_mem, img_height * img_width * sizeof(float)));

	GetDispFloat << <128, 128 >> > (d_rightDispFloat_mem, d_minCosts_mem, d_minDispValue_mem, sub_pixel, img_height, img_width, max_disp);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
	checkCudaErrors(cudaFree(d_minCosts_mem));
	checkCudaErrors(cudaFree(d_minDispValue_mem));

	cudaMalloc(&d_left2rightDisp_mem, img_height * img_width * sizeof(float));
	cudaMalloc(&d_right2leftDisp_mem, img_height * img_width * sizeof(float));

	GetDispInterp << <128, 128 >> > (d_right2leftDisp_mem, d_rightDispFloat_mem, d_leftDispFloat_mem, 1, img_height, img_width);
	GetDispInterp << <128, 128 >> > (d_left2rightDisp_mem, d_leftDispFloat_mem, d_rightDispFloat_mem, -1, img_height, img_width);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());

	ThresholdDisp << <128, 128 >> > (d_leftDispFloat_mem, d_right2leftDisp_mem, threshold, img_height, img_width);
	ThresholdDisp << <128, 128 >> > (d_rightDispFloat_mem, d_left2rightDisp_mem, threshold, img_height, img_width);
	cudaThreadSynchronize();
	checkCUDA(__LINE__, cudaGetLastError());
	checkCudaErrors(cudaFree(d_left2rightDisp_mem)); checkCudaErrors(cudaFree(d_right2leftDisp_mem));


	checkCudaErrors(cudaMalloc(&d_leftDispUchar_mem, img_height * img_width * sizeof(uchar)));
	checkCudaErrors(cudaMalloc(&d_rightDispUchar_mem, img_height * img_width * sizeof(uchar)));

	GetDispUchar << <128, 128 >> > (d_leftDispUchar_mem, d_leftDispFloat_mem, img_height, img_width, max_disp);
	GetDispUchar << <128, 128 >> > (d_rightDispUchar_mem, d_rightDispFloat_mem, img_height, img_width, max_disp);
	cudaThreadSynchronize();
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