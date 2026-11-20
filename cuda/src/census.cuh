#ifndef CENSUS_CUH
#define CENSUS_CUH

// Shared declarations for the three CUDA stages (v1_baseline.cu, v2_fused.cu,
// v3_rolling.cu). Exactly one of them is linked into each executable, so all
// three define the same DiffBoxImages() symbol.

#include <opencv2/core.hpp>  // uchar / ushort / uint typedefs (no `using namespace` here)
#include <cuda_runtime.h>
#include "helper_cuda.h"

// Launch configuration used for every kernel. All kernels use grid-stride
// loops, so any (blocks, threads) pair is correct; these values are only a
// tuning choice.
constexpr int kNumBlocks  = 128;
constexpr int kNumThreads = 128;

// Runs the whole stereo pipeline on the GPU.
//   pLeftOutPfm / pRightOutPfm : float disparity maps, img_height*img_width, filled on return
//   pLeftOutImg / pRightOutImg : uchar visualisation (disparity * 256/max_disp, clamped to 255)
//   pimageLeft / pimageRight   : input 8-bit grayscale images, row-major, img_height*img_width
// All pointers are host memory and must be pre-allocated by the caller.
void DiffBoxImages(float* pLeftOutPfm, float* pRightOutPfm, uchar* pLeftOutImg, uchar* pRightOutImg, uchar* pimageLeft, uchar* pimageRight,
	const bool sub_pixel, const int threshold, const int census_filter_size, const int aggreate_filter_size,
	const int img_height, const int img_width, const int max_disp);
void checkCUDA(const int lineNumber, cudaError_t status);
void FatalError(const int lineNumber);

#endif  // CENSUS_CUH
