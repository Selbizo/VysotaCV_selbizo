#include "StGpu.h"
#include "measuring.hpp"

/* Write results to files in the directory */
void writeRes(Mat TStab, Point2f d, string dirName)
{
	// - - - - - - - Outputing - - - - - - - 

	string pathDir = "./" + dirName + "/";

	mkdir(dirName.c_str(), 0777); // 0777 sets permissions
    
    // Check if directory exists
    struct stat info;
    if (stat(dirName.c_str(), &info) != 0) {
        std::cerr << "Cannot access directory\n";
    } 

	// if (filesystem::create_directories(pathDir)) {
    //     std::cout << "Directory created: " << pathDir << std::endl;
    // }

	// ofstream xFile(pathDir + "x.txt", ios::app);
	// ofstream yFile(pathDir + "y.txt", ios::app);
	// ofstream angFile(pathDir + "ang.txt", ios::app);
	ofstream dxFile(pathDir + "dx.txt", ios::app);
	ofstream dyFile(pathDir + "dy.txt", ios::app);

	// // output x
	// if (xFile){
	// 	xFile << TStab.at<float>(0, 2) << endl;
	// }
	
	// // output y
	// if (yFile){
	// 	yFile << TStab.at<float>(1, 2) << endl;
	// }

	// // output angle
	// if (angFile){
	// 	angFile << 180.0 / 3.14 * atan2(TStab.at<float>(1, 0), TStab.at<float>(0, 0)) << endl;
	// }

	// output dx
	if (dxFile){
		dxFile << d.x << endl;
	}

	// output dy
	if (dyFile){
		dyFile << d.y << endl;
	}
}

/* Measuring image deviation */
void measuring(
    cuda::GpuMat oldFrame, 
    cuda::GpuMat nextFrame, 
    Ptr<cuda::CornersDetector> detector, 
    double tauStab, 
    double kSwitch, 
    Rect roi,
    string name_dir="meas"
)
{

	int n = 2;
	const double cols = oldFrame.cols;
	const double rows = oldFrame.rows;
	const double diag = sqrt(cols * cols + rows * rows);
	const double atang = atan2(rows, cols);

	// deep copy for iirAdaptive
	double tau = tauStab;
	double kSw = kSwitch;
	Rect ROI = roi;

	vector<Point2f> p0, p1;
	cuda::GpuMat gP0, gP1;
	cuda::GpuMat gStatus;

	cuda::GpuMat nextGray;
	cuda::GpuMat oldGray;

	Ptr<cuda::SparsePyrLKOpticalFlow> optflow = cuda::SparsePyrLKOpticalFlow::create(
		Size(25, 25), 
		3, 
		3
	);

	// vector<uchar> status_meas;
	Mat T;
	Mat TStab(2, 3, CV_64F);
	Point2f d = Point2f(0.0f, 0.0f);
	vector<TransformParam> transforms(2);
	for (int i = 0; i < 2; i++)
	{
		transforms[i].dx = 0.0;
		transforms[i].dy = 0.0;
		transforms[i].da = 0.0;
	}

	cuda::resize(oldFrame, oldGray, Size(cols / 2, rows / 2), 0.0, 0.0, INTER_LINEAR);
	cuda::resize(nextFrame, nextGray, Size(cols / 2, rows / 2), 0.0, 0.0, INTER_LINEAR);

	cuda::cvtColor(oldGray, oldGray, COLOR_BGR2GRAY);
	cuda::cvtColor(nextGray, nextGray, COLOR_BGR2GRAY);

	detector->detect(oldGray, gP0);
	optflow->calc(oldGray, nextGray, gP0, gP1, gStatus);

	gP1.download(p1);
	gP0.download(p0);
	// custom_download(gStatus_meas, status_meas);

	getBiasAndRotation(p0, p1, d, transforms, T, n);
	iirAdaptive(transforms, tau, ROI, cols, rows, kSw);

	transforms[1].getTransform(TStab, cols, rows, diag, atang);

	writeRes(TStab, d, name_dir);
}

