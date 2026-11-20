// Command-line driver shared by census_v1 / census_v2 / census_v3.
// Exactly one of v1_baseline.cu, v2_fused.cu, v3_rolling.cu is linked in, and all of them
// export the same DiffBoxImages() function declared in census.cuh.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "census.cuh"
#include "pfm_io.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;

struct CudaDevice {
	int device_id;
	int pci_id;
	string name;
};

static void printUsage(const char* prog)
{
	printf(
		"Usage: %s LEFT_IMAGE RIGHT_IMAGE [options]\n"
		"\n"
		"Census-transform stereo matching on the GPU (8-bit grayscale input).\n"
		"LEFT_IMAGE is the reference image: its pixel (x, y) is matched with RIGHT_IMAGE at\n"
		"(x + d, y), d >= 0. For a standard rectified pair this means LEFT_IMAGE must be the\n"
		"picture taken by the physically RIGHT camera (swap the files if the result looks empty).\n"
		"\n"
		"Options (defaults in brackets):\n"
		"  --census N       census window size, odd              [11]\n"
		"  --aggregate N    box-filter window size, odd          [11]\n"
		"  --max-disp N     number of disparities searched       [300]\n"
		"  --sub-pixel 0|1  parabolic sub-pixel refinement       [1]\n"
		"  --threshold N    left-right check tolerance (pixels)  [10]\n"
		"  --device N       index into the list of CUDA devices  [0]\n"
		"  --out-dir DIR    directory for left/right .png/.pfm   [.]\n"
		"  --no-display     do not open preview windows (headless)\n"
		"  --verbose        print CUDA device properties\n"
		"  -h, --help       show this message\n"
		"\n"
		"Outputs: left.png, right.png (8-bit visualisation), left.pfm, right.pfm (float disparity).\n",
		prog);
}

static bool parseInt(const char* text, int min_value, int& out)
{
	char* end = NULL;
	errno = 0;
	long v = strtol(text, &end, 10);
	if (errno != 0 || end == text || *end != '\0' || v < min_value || v > INT_MAX)
		return false;
	out = (int)v;
	return true;
}

static void printDeviceInfo()
{
	int count = 0;
	checkCudaErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop, i));
		printf("   --- General Information for device %d ---\n", i);
		printf("Name:  %s\n", prop.name);
		printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf("Kernel execution timeout :  %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");
		printf("\n");

		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem:  %zu\n", (size_t)prop.totalGlobalMem);
		printf("Total constant Mem:  %zu\n", (size_t)prop.totalConstMem);
		printf("Max mem pitch:  %zu\n", (size_t)prop.memPitch);
		printf("Texture Alignment:  %zu\n", (size_t)prop.textureAlignment);
		printf("\n");

		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
		printf("Shared mem per block:  %zu\n", (size_t)prop.sharedMemPerBlock);
		printf("Registers per block:  %d\n", prop.regsPerBlock);
		printf("Threads in warp:  %d\n", prop.warpSize);
		printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}
}

// Lists CUDA devices of compute capability >= 3.0 sorted by PCI bus id and selects the
// `index`-th one (so index 0 is the lowest PCI bus id, not necessarily CUDA device 0).
static void selectCudaDevice(int index)
{
	int deviceCount = 0;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "There is no cuda capable device!\n");
		exit(EXIT_FAILURE);
	}
	cout << "Detected " << deviceCount << " devices!" << endl;

	std::vector<CudaDevice> usableDevices;
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 3) {
				CudaDevice dev;
				dev.device_id = i;
				dev.pci_id = prop.pciBusID;
				dev.name = prop.name;
				usableDevices.push_back(dev);
			}
			else {
				cout << "CUDA capable device " << string(prop.name)
					<< " is only compute capability " << prop.major << '.'
					<< prop.minor << endl;
			}
		}
		else {
			cout << "Could not check device properties for one of the cuda devices!" << endl;
		}
	}

	if (usableDevices.empty()) {
		fprintf(stderr, "There is no usable cuda device (compute capability >= 3.0 required)!\n");
		exit(EXIT_FAILURE);
	}

	std::sort(usableDevices.begin(), usableDevices.end(),
		[](const CudaDevice& a, const CudaDevice& b) { return a.pci_id < b.pci_id; });

	for (size_t i = 0; i < usableDevices.size(); i++) {
		printf("%02d : %s\n", (int)i, usableDevices[i].name.c_str());
	}

	if (index < 0 || index >= (int)usableDevices.size()) {
		fprintf(stderr, "Error: --device %d is out of range (0..%d).\n", index, (int)usableDevices.size() - 1);
		exit(EXIT_FAILURE);
	}

	cout << "Using CUDA device: " << usableDevices[index].name << endl;
	checkCudaErrors(cudaSetDevice(usableDevices[index].device_id));
}

int main(int argc, char* argv[])
{
	// Defaults.
	int device_index = 0;
	int census_filter_size = 11;
	int aggreate_filter_size = 11;
	int max_disp = 300;
	bool sub_pixel = true;
	int threshold = 10;
	string out_dir = ".";
	bool display = true;
	bool verbose = false;
	std::vector<string> positional;

	// Argument parsing.
	for (int a = 1; a < argc; a++) {
		const string arg = argv[a];
		const bool has_value = (a + 1 < argc);
		int value = 0;

		if (arg == "-h" || arg == "--help") {
			printUsage(argv[0]);
			return 0;
		}
		else if (arg == "--no-display") { display = false; }
		else if (arg == "--verbose") { verbose = true; }
		else if (arg == "--out-dir") {
			if (!has_value) { fprintf(stderr, "Error: --out-dir needs a value.\n"); return 1; }
			out_dir = argv[++a];
		}
		else if (arg == "--census" || arg == "--aggregate" || arg == "--max-disp" ||
			arg == "--sub-pixel" || arg == "--threshold" || arg == "--device") {
			if (!has_value) { fprintf(stderr, "Error: %s needs a value.\n", arg.c_str()); return 1; }
			const char* text = argv[++a];
			const int min_value = (arg == "--census" || arg == "--aggregate" || arg == "--max-disp") ? 1 : 0;
			if (!parseInt(text, min_value, value) || (arg == "--sub-pixel" && value > 1)) {
				fprintf(stderr, "Error: invalid value '%s' for %s.\n", text, arg.c_str());
				return 1;
			}
			if (arg == "--census") census_filter_size = value;
			else if (arg == "--aggregate") aggreate_filter_size = value;
			else if (arg == "--max-disp") max_disp = value;
			else if (arg == "--sub-pixel") sub_pixel = (value != 0);
			else if (arg == "--threshold") threshold = value;
			else device_index = value;
		}
		else if (arg.size() > 1 && arg[0] == '-') {
			fprintf(stderr, "Error: unknown option '%s'. Try --help.\n", arg.c_str());
			return 1;
		}
		else {
			positional.push_back(arg);
		}
	}

	if (positional.size() != 2) {
		fprintf(stderr, "Error: expected LEFT_IMAGE and RIGHT_IMAGE.\n\n");
		printUsage(argv[0]);
		return 1;
	}
	const string Limg = positional[0];
	const string Rimg = positional[1];

	if (census_filter_size % 2 == 0 || aggreate_filter_size % 2 == 0)
		fprintf(stderr, "Warning: window sizes are expected to be odd (even sizes give an off-centre window).\n");

	cv::Mat imageLeft = cv::imread(Limg, cv::IMREAD_GRAYSCALE);
	cv::Mat imageRight = cv::imread(Rimg, cv::IMREAD_GRAYSCALE);
	if (imageLeft.empty()) {
		fprintf(stderr, "Error: cannot read left image '%s'.\n", Limg.c_str());
		return 1;
	}
	if (imageRight.empty()) {
		fprintf(stderr, "Error: cannot read right image '%s'.\n", Rimg.c_str());
		return 1;
	}
	if (imageLeft.size() != imageRight.size()) {
		fprintf(stderr, "Error: image sizes differ (%dx%d vs %dx%d).\n",
			imageLeft.cols, imageLeft.rows, imageRight.cols, imageRight.rows);
		return 1;
	}

	if (verbose) printDeviceInfo();
	selectCudaDevice(device_index);

	const int img_height = imageLeft.rows;
	const int img_width = imageLeft.cols;

	cv::Mat leftOutImg(imageLeft.size(), CV_8UC1);
	cv::Mat rightOutImg(imageLeft.size(), CV_8UC1);
	cv::Mat leftOutPfm(imageLeft.size(), CV_32FC1);
	cv::Mat rightOutPfm(imageLeft.size(), CV_32FC1);

	clock_t start = clock();
	DiffBoxImages((float*)leftOutPfm.ptr(), (float*)rightOutPfm.ptr(), (uchar*)leftOutImg.ptr(), (uchar*)rightOutImg.ptr(),
		(uchar*)imageLeft.ptr(), (uchar*)imageRight.ptr(), sub_pixel, threshold, census_filter_size, aggreate_filter_size,
		img_height, img_width, max_disp);
	clock_t end = clock();

	printf("computation time: %f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

	const string prefix = out_dir + "/";
	bool ok = true;
	ok &= cv::imwrite(prefix + "left.png", leftOutImg);
	ok &= cv::imwrite(prefix + "right.png", rightOutImg);
	ok &= savePFM(leftOutPfm, prefix + "left.pfm");
	ok &= savePFM(rightOutPfm, prefix + "right.pfm");
	if (!ok) {
		fprintf(stderr, "Error: could not write all outputs to '%s' (does the directory exist?).\n", out_dir.c_str());
		return 1;
	}

	if (display) {
		cv::namedWindow("left output", cv::WINDOW_NORMAL);
		cv::imshow("left output", leftOutImg);
		cv::namedWindow("right output", cv::WINDOW_NORMAL);
		cv::imshow("right output", rightOutImg);
		cv::waitKey(0);
	}

	return 0;
}
