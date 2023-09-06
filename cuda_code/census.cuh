#include <opencv2/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
using namespace cv;
#ifndef census_h
#define census_h

void DiffBoxImages(float* pLeftOutPfm, float* pRightOutPfm, uchar* pLeftOutImg, uchar* pRightOutImg, uchar* pimageLeft, uchar* pimageRight,
	const bool sub_pixel, const int threshold, const int census_filter_size, const int aggreate_filter_size,
	const int img_height, const int img_width, const int max_disp);
void checkCUDA(const int lineNumber, cudaError_t status);
void FatalError(const int lineNumber);
#endif