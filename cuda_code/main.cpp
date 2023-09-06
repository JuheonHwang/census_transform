#include <opencv2/highgui.hpp>
#include "census.cuh"
#include "helper_cuda.h"
#include "PFMReadWrite.h"
#include <opencv2/imgproc.hpp>

#include <string>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;



struct CUDA_DEVICE {
	int device_id;
	int pci_id;
	string name;
};

int compare(const void* a, const void* b) {
	return ((CUDA_DEVICE*)a)->pci_id - ((CUDA_DEVICE*)b)->pci_id;
}

static void selectCudaDevice(int pci_id)
{
	int deviceCount = 0;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "There is no cuda capable device!\n");
		exit(EXIT_FAILURE);
	}
	cout << "Detected " << deviceCount << " devices!" << endl;

	std::vector<CUDA_DEVICE> usableDevices;


	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 3 && prop.minor >= 0) {
				CUDA_DEVICE dev;
				dev.device_id = i;
				dev.pci_id = prop.pciBusID;
				dev.name = prop.name;
				usableDevices.push_back(dev);
			}
			else {
				cout << "CUDA capable device " << string(prop.name)
					<< " is only compute cabability " << prop.major << '.'
					<< prop.minor << endl;
			}
		}
		else {
			cout << "Could not check device properties for one of the cuda "
				"devices!" << endl;
		}
	}


	if (usableDevices.empty()) {
		fprintf(stderr, "There is no cuda device supporting gipuma!\n");
		exit(EXIT_FAILURE);
	}

	// usableDevices 순회를 해보세요.



	// PCI ID에 대해서 소팅

	qsort(data(usableDevices), usableDevices.size(), sizeof(usableDevices[0]), compare);

	// usableDevices 순회를 해보세요.

	int i = 0;
	for (vector<CUDA_DEVICE>::iterator iter = usableDevices.begin(); iter != usableDevices.end(); iter++, i++) {
		printf("%02d : %s\n", i, iter->name.c_str());
	}

	cout << "Detected gipuma compatible device: " << usableDevices[pci_id].name << endl;;
	checkCudaErrors(cudaSetDevice(usableDevices[pci_id].device_id));

	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 128);
}

int main(int argc, char* argv[])
{

	int pci_id = 0;


	//string Limg = "images/TEST_RealSense_dot/R_infrared_0002_06.png";
	//string Rimg = "images/TEST_RealSense_dot/L_infrared_0001_06.png";
	//string Limg = "images/TEST_RealSense_dot/R_infrared_0002_01.png";
	//string Rimg = "images/TEST_RealSense_dot/L_infrared_0001_01.png";
	//string Limg = "images/TEST_RealSense/L_infrared_0001_01.png";
	//string Rimg = "images/TEST_RealSense/R_infrared_0002_01.png";
	//string Limg = "images/TEST_Viewworks/L_infrared_0001_01.png";
	//string Rimg = "images/TEST_Viewworks/R_infrared_0002_01.png";
	//string Limg = "images/220111/R5.png";
	//string Rimg = "images/220111/L5.png";
	//string Limg = "images/R_infrared_0002_02.png";
	//string Rimg = "images/L_infrared_0001_02.png";
	//string Limg = "images/R_infrared_0002_03.png";
	//string Rimg = "images/L_infrared_0001_03.png";
	//string Limg = "images/220507_hjh/R3.png";
	//string Rimg = "images/220507_hjh/L3.png";
	//string Limg = "images/220609_for_projector_test/test_b.png";
	//string Rimg = "images/220609_for_projector_test/test_a.png";
	string Limg = "images/Blender_0616/im2.png";
	string Rimg = "images/Blender_0616/im1.png";


	//string Limg = "R_infrared_0002_01.png";
	//string Rimg = "L_infrared_0001_01.png";
	//string Limg = "R_infrared_0002_02.png";
	//string Rimg = "L_infrared_0001_02.png";
	//string Limg = "R_infrared_0002_03.png";
	//string Rimg = "L_infrared_0001_03.png";
	//string Limg = "R_infrared_0002_04.png";
	//string Rimg = "L_infrared_0001_04.png";
	//string Limg = "L.png";
	//string Rimg = "R.png";

	//int census_filter_size = 7;
	//int aggreate_filter_size = 11;
	//int max_disp = 80;
	//bool sub_pixel = true;
	//int threshold = 5;
	
	//int census_filter_size = 15;
	//int aggreate_filter_size = 19;
	//int max_disp = 96;
	//bool sub_pixel = true;
	//int threshold = 10;

	//const int census_filter_size = 37;
	//const int aggreate_filter_size = 45;
	//const int max_disp = 300;
	//const bool sub_pixel = true;
	//const int threshold = 10;

	//const int census_filter_size = 25;
	//const int aggreate_filter_size = 31;
	//const int max_disp = 1024;
	//const bool sub_pixel = true;
	//const int threshold = 10;

	//const int census_filter_size = 13;
	//const int aggreate_filter_size = 13;
	//const int max_disp = 320;
	//const bool sub_pixel = true;
	//const int threshold = 10;

	const int census_filter_size = 11;
	const int aggreate_filter_size = 11;
	const int max_disp = 300;
	const bool sub_pixel = true;
	const int threshold = 10;

	//const int census_filter_size = 27;
	//const int aggreate_filter_size = 35;
	//const int max_disp = 1024;
	//const bool sub_pixel = true;
	//const int threshold = 10;

	//if (argc > 8) {
	//	pci_id = atoi(argv[1]);
	//	Limg = argv[2];
	//	Rimg = argv[3];
	//	census_filter_size = atoi(argv[4]);
	//	aggreate_filter_size = atoi(argv[5]);
	//	max_disp = atoi(argv[6]);
	//	sub_pixel = argv[7] > 0;
	//	threshold = atoi(argv[8]);
	//}
	//else {
	//	return 1;
	//}

	Mat imageLeft = imread(Limg.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageRight = imread(Rimg.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	//resize(imageLeft, imageLeft, Size(imageLeft.cols * 0.25, imageLeft.rows * 0.25), 0, 0, CV_INTER_LINEAR);
	//resize(imageRight, imageRight, Size(imageRight.cols * 0.25, imageRight.rows * 0.25), 0, 0, CV_INTER_LINEAR);
	//resize(imageLeft, imageLeft, Size(2560, 2560), 0, 0, CV_INTER_LINEAR);
	//resize(imageRight, imageRight, Size(2560, 2560), 0, 0, CV_INTER_LINEAR);
	//resize(imageLeft, imageLeft, Size(480, 480), 0, 0, CV_INTER_LINEAR);
	//resize(imageRight, imageRight, Size(480, 480), 0, 0, CV_INTER_LINEAR);
	//resize(imageLeft, imageLeft, Size(318, 180), 0, 0, CV_INTER_LINEAR);
	//resize(imageRight, imageRight, Size(318, 180), 0, 0, CV_INTER_LINEAR);

	selectCudaDevice(pci_id);

	//Mat leftOutput(size(imageLeft), CV_16U);
	//Mat rightOutput(size(imageLeft), CV_16U);

	//imshow("left image", imageLeft);
	//waitKey(0);
	//imshow("right image", imageRight);
	//waitKey(0);



	//resize(imageLeft, imageLeft, Size(imageLeft.cols * 0.5, imageLeft.rows * 0.5), 0, 0, CV_INTER_LINEAR);
	//resize(imageRight, imageRight, Size(imageRight.cols * 0.5, imageRight.rows * 0.5), 0, 0, CV_INTER_LINEAR);

	const int img_height = imageLeft.rows;
	const int img_width = imageLeft.cols;

	Mat leftOutImg(imageLeft.clone());
	Mat rightOutImg(imageLeft.clone());
	Mat leftOutPfm(imageLeft.clone());
	leftOutPfm.convertTo(leftOutPfm, CV_32F);
	Mat rightOutPfm(imageLeft.clone());
	rightOutPfm.convertTo(rightOutPfm, CV_32F);

	clock_t start = clock();
	DiffBoxImages((float*)leftOutPfm.ptr(), (float*)rightOutPfm.ptr(),  (uchar*)leftOutImg.ptr(), (uchar*)rightOutImg.ptr(), (uchar*)imageLeft.ptr(), (uchar*)imageRight.ptr(), sub_pixel, threshold, census_filter_size, aggreate_filter_size, img_height, img_width, max_disp);
	clock_t end = clock();

	printf("computation time: %f sec\n", (double)(end - start)/1000);

	namedWindow("left output", WINDOW_NORMAL);
	//resizeWindow("left output", 1280, 1280);
	//resizeWindow("left output", 1280, 720);
	imshow("left output", leftOutImg);
	//waitKey(0);

	namedWindow("right output", WINDOW_NORMAL);
	//resizeWindow("right output", 1280, 1280);
	//resizeWindow("right output", 1280, 720);
	imshow("right output", rightOutImg);
	waitKey(0);

	imwrite("left.png", leftOutImg);
	imwrite("right.png", rightOutImg);
	// savePFM(leftOutPfm, "left.pfm");
	// savePFM(rightOutPfm, "right.pfm");

	//Mat leftOutputShow;
	//leftOutput.convertTo(leftOutputShow, CV_8U, 1);
	//imshow("left output", leftOutputShow * 2);
	//waitKey(0);
	//
	//Mat rightOutputShow;
	//rightOutput.convertTo(rightOutputShow, CV_8U, 1);
	//imshow("right output", rightOutputShow * 2);
	//waitKey(0);

	return 0;
}