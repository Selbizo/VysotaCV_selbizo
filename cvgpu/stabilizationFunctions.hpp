// Здесь представлены функции, отвечающие за электронную стабилизацию видеопотока
#pragma once

// Основные заголовки OpenCV
#include <opencv2/core.hpp>          // Базовые структуры (Mat, Point2f)
#include <opencv2/imgproc.hpp>       // Операции с изображениями
#include <opencv2/videoio.hpp>       // VideoCapture
#include <opencv2/core/cuda.hpp>     // CUDA-функционал (GpuMat)
#include <opencv2/cudaarithm.hpp>    // CUDA-арифметика
#include <opencv2/cudaimgproc.hpp>   // CUDA-операции с изображениями
#include <opencv2/calib3d.hpp>       // estimateAffine2D

// Стандартные заголовки C++
#include <vector>    // std::vector
#include <iostream>  // std::cout

using namespace cv;
using namespace std;

void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldCompressed, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gOldCompressed, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d,
	vector <TransformParam>& transforms, Mat& T, const int compression);

void iir(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, Mat& frame);

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, const int a, const int b, const double c, double& kSwitch, vector<TransformParam>& velocity);

void iirAdaptiveHighPass(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, int cols, int rows, double& kSwitch);



void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldCompressed, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible)
{
	capture >> oldFrame;

	gOldFrame.upload(oldFrame);
	gOldCompressed.release();
	cuda::resize(gOldFrame, gOldCompressed, Size(a / compression, b / compression), 0.0, 0.0, cv::INTER_LINEAR);
	cuda::cvtColor(gOldCompressed, gOldGray, COLOR_BGR2GRAY);
	cuda::bilateralFilter(gOldGray, gOldGray, 3, 3.0, 1.0); //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols / frame compression , gOldGray.rows / frame compression ), 0.0, 0.0, cv::INTER_AREA);

	if (qualityLevel > 0.001 && harrisK > 0.001)
	{
		qualityLevel *= 0.6;
		harrisK *= 0.6;
	}
	else
	{
		if (maxCorners > 50)
		{
			maxCorners *= 0.98;
			d_features->setMaxCorners(maxCorners);
		}
	}
	for (int i = 0; i < 1;i++)
	{
		transforms[i].dx *= kSwitch;
		transforms[i].dy *= kSwitch;
		transforms[i].da *= kSwitch;
	}

	d_features->detect(gOldGray, gP0, mask_device);

	if ((gP0.cols > 20)) {

		p0.clear();
		gP0.download(p0);
		stab_possible = true; //true
	}
	else {
		stab_possible = false;
	}
}


void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gOldCompressed, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible)
{
	gOldFrame.upload(oldFrame);
	cuda::resize(gOldFrame, gOldCompressed, Size(a / compression, b / compression), 0.0, 0.0, cv::INTER_AREA);

	cuda::cvtColor(gOldCompressed, gOldGray, COLOR_BGR2GRAY);
	cuda::bilateralFilter(gOldGray, gOldGray, 3, 3.0, 3.0);
	//cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols, gOldGray.rows), 0.0, 0.0, cv::INTER_AREA);
	stab_possible = false;
}


void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, 
	vector <TransformParam>& transforms, Mat& T, const int compression)
{
	const double N = 1.0;
	for (uint i = 0; i < p1.size(); i++)
	{
		if (i == 0)
		{
			d = p1[0] - p0[0];
		}
		d = d + (p1[i] - p0[i]);
	}

	d = d * compression / (int)p0.size();

	if (p0.empty() || p1.empty() || (p1.size() != p0.size()))
	{
		transforms[0] = TransformParam(-d.x * compression, -d.y * compression, 0.0);
		cout << "bull shit" << endl;
	}
	else
	{
		T = estimateAffine2D(p0, p1);
		transforms[0] = TransformParam(-(T.at<double>(0, 2) * N + d.x * (1.0 - N)), -(T.at<double>(1, 2) * N + d.y * (1.0 - N)), -atan2(T.at<double>(1, 0), T.at<double>(0, 0)));
	}
}


void iir(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, Mat& frame)
{

	transforms[1].dx = transforms[1].dx * (tau_stab - 1.0) / tau_stab + transforms[0].dx;
	transforms[1].dy = transforms[1].dy * (tau_stab - 1.0) / tau_stab + transforms[0].dy;
	transforms[1].da = transforms[1].da * (0.5 * tau_stab - 1.0) / (0.5 * tau_stab) + transforms[0].da;
	if (transforms[1].da > 1.0)
	{
		transforms[1].da = 1.0;
	}
	if (transforms[1].da < -1.0)
	{
		transforms[1].da = -1.0;
	}
	if (tau_stab < 150.0) {
		tau_stab *= 1.1;
	}
	if (tau_stab < 500.0 && !(abs(transforms[1].dx) > 15.0 || abs(transforms[1].dy) > 15.0)) {
		tau_stab *= 1.1;
	}
	if (roi.x + (int)transforms[1].dx < 0)
	{
		transforms[1].dx = double(1 - roi.x);
		if (tau_stab > 50) {
			tau_stab *= 0.8;
			transforms[1].da *= 0.95;
		}

	}
	else if (roi.x + roi.width + (int)transforms[1].dx >= frame.cols)
	{
		transforms[1].dx = (double)(frame.cols - roi.x - roi.width);
		if (tau_stab > 50) {
			tau_stab *= 0.8;
			transforms[1].da *= 0.95;
		}
	}

	if (roi.y + (int)transforms[1].dy < 0)
	{
		transforms[1].dy = (double)(1 - roi.y);
		if (tau_stab > 50) {
			tau_stab *= 0.8;
			transforms[1].da *= 0.95;
		}
	}
	else if (roi.y + roi.height + (int)transforms[1].dy >= frame.rows)
	{
		transforms[1].dy = (double)(frame.rows - roi.y - roi.height);
		if (tau_stab > 50) {
			tau_stab *= 0.8;
			transforms[1].da *= 0.95;
		}
	}
}

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, const int a, const int b, const double c, double& kSwitch, vector<TransformParam>& movement)//, cv::KalmanFilter& KF)
{
	//if ((abs(transforms[0].dx) - 10.0 < 1.2 * transforms[3].dx) && (abs(transforms[0].dy) - 10.0 < 1.2 * transforms[3].dy) && (abs(transforms[0].da) - 0.02 < 1.2 * transforms[3].da))//проверка на выброс и предельно минимальную амплитуду отклонения
	if ((abs(transforms[0].dx) - 10.0 < 3.0 * transforms[3].dx) && (abs(transforms[0].dy) - 10.0 < 3.0 * transforms[3].dy) && (abs(transforms[0].da) - 0.02 < 3.0 * transforms[3].da))//проверка на выброс и предельно минимальную амплитуду отклонения
	{
		transforms[1].dx = kSwitch * (transforms[1].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[0].dx); //накопление по перемещнию внутри кадра
		transforms[1].dy = kSwitch * (transforms[1].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[0].dy);
		transforms[1].da = kSwitch * (transforms[1].da * (tau_stab - 4.0) / tau_stab + kSwitch * transforms[0].da);
	}

	if (transforms[1].da > 3.0)
		transforms[1].da = 3.0;

	if (transforms[1].da < -3.0)
		transforms[1].da = -3.0;

	if (tau_stab < 30.0)
		tau_stab *= 1.1;

	if (tau_stab < 50.0 && !(abs(transforms[1].dx) > a / 2 || abs(transforms[1].dy) > b / 2))
		tau_stab *= 1.1;

	if (tau_stab < 120.0 && !(abs(transforms[1].dx) > a / 3 || abs(transforms[1].dy) > b / 3))
	{
		tau_stab *= 1.1;
		if (tau_stab > 120.0)
			tau_stab = 120.0;
	}


	if (roi.x + (int)transforms[1].dx < 0)
	{
		transforms[1].dx = double(1 - roi.x);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[1].da *= 0.999;
			kSwitch *= 0.95;
		}
		//cout << "-> right border collision" << endl;
	}
	else if (roi.x + roi.width + (int)transforms[1].dx >= a)
	{
		transforms[1].dx = (double)(a - roi.x - roi.width);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[1].da *= 0.999;
			kSwitch *= 0.95;
		}
		//cout << "<- left border collision" << endl;
	}

	if (roi.y + (int)transforms[1].dy < 0)
	{
		transforms[1].dy = (double)(1 - roi.y);
		if (tau_stab > 10) {
			tau_stab *= 0.9;
			transforms[1].da *= 0.999;
			kSwitch *= 0.95;
		}
		//cout << "down border collision" << endl;
	}
	else if (roi.y + roi.height + (int)transforms[1].dy >= b)
	{
		transforms[1].dy = (double)(b - roi.y - roi.height);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[1].da *= 0.999;
			kSwitch *= 0.95;
		}
		//cout << "uppper border collision" << endl;
	}

	if (kSwitch < 1.0)
		tau_stab *= (4.0 + kSwitch) / 5.0;

	//transforms[2].dx = (1.0 - 0.05) * transforms[2].dx + 0.05 * abs(transforms[1].dx);
	//transforms[2].dy = (1.0 - 0.05) * transforms[2].dy + 0.05 * abs(transforms[1].dy);
	//transforms[2].da = (1.0 - 0.05) * transforms[2].da + 0.05 * abs(transforms[1].da);

	//if ((abs(transforms[0].dx) - 10.0 < 2.2 * transforms[3].dx || abs(transforms[0].dx) < 0.0) && (abs(transforms[0].dy) - 10.0 < 2.2 * transforms[3].dy || abs(transforms[0].dy) < 0.0) && (abs(transforms[0].da) - 0.01 < 2.2 * transforms[3].da || abs(transforms[0].da) < 0.0))
	if (true)
	{
		transforms[3].dx = (1.0 - 0.9) * transforms[3].dx + 0.9 * abs(transforms[0].dx); //среднее абсолютное отклонение колебаний
		transforms[3].dy = (1.0 - 0.9) * transforms[3].dy + 0.9 * abs(transforms[0].dy);
		transforms[3].da = (1.0 - 0.9) * transforms[3].da + 0.9 * abs(transforms[0].da);
	}



	movement[2].dx = (movement[2].dx*3 + movement[1].dx - transforms[0].dx)/4; //velocities first derivative 
	movement[2].dy = (movement[2].dy*3 + movement[1].dy - transforms[0].dy)/4; //velocities first derivative
	movement[2].da = (movement[2].da*3 + movement[1].da - transforms[0].da)/4; //velocities first derivative

	movement[1].dx = (movement[1].dx*127 + transforms[0].dx)/128; //velocities first derivative 
	movement[1].dy = (movement[1].dy*127 + transforms[0].dy)/128; //velocities first derivative
	movement[1].da = (movement[1].da*127 + transforms[0].da)/128; //velocities first derivative

	movement[0].dy = movement[1].dy + movement[0].dy*127/128; //coordinates
	movement[0].da = movement[1].da + movement[0].da*127/128; //coordinates
	movement[0].dx = movement[1].dx + movement[0].dx*127/128; //coordinates

	transforms[2].dx = transforms[1].dx - movement[1].dx; //coordinates
	transforms[2].dy = transforms[1].dy - movement[1].dy; //coordinates
	transforms[2].da = transforms[1].da - movement[1].da; //coordinates



}

void iirAdaptiveHighPass(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, int cols, int rows, double& kSwitch)//, cv::KalmanFilter& KF)
{
	if ((abs(transforms[0].dx) - (double)rows / 2 < 1.2 * transforms[3].dx) && (abs(transforms[0].dy) - (double)rows / 2 < 1.2 * transforms[3].dy) && (abs(transforms[0].da) - 0.2 < 3.0 * transforms[3].da))//проверка на выброс и предельно минимальную амплитуду отклонения
	{
		transforms[1].dx = kSwitch * (transforms[1].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[0].dx);
		transforms[1].dy = kSwitch * (transforms[1].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[0].dy);
		transforms[1].da = kSwitch * (transforms[1].da * (1.7 * tau_stab - 1.0) / (1.7 * tau_stab) + kSwitch * transforms[0].da);
	}

	if (transforms[1].da > 1.4)
	{
		transforms[1].da = 1.4;
	}
	if (transforms[1].da < -1.4)
	{
		transforms[1].da = -1.4;
	}
	if (tau_stab < 10.0) {
		tau_stab *= 1.1;
	}
	if (tau_stab < 20.0 && !(abs(transforms[1].dx) > 60.0 || abs(transforms[1].dy) > 60.0)) {
		tau_stab *= 1.1;
	}
	if (tau_stab < 30.0 && !(abs(transforms[1].dx) > 30.0 || abs(transforms[1].dy) > 30.0)) {
		tau_stab *= 1.1;
		if (tau_stab > 30.0)
			tau_stab = 30.0;
	}
	if (roi.x + (int)transforms[1].dx < 0)
	{
		transforms[1].dx = double(1 - roi.x);
		if (tau_stab > 50) {
			tau_stab *= 0.9;

			//transforms[1].dx -= 0.95 * transforms[0].dx;
			//transforms[1].dy -= 0.95 * transforms[0].dy;
			//transforms[1].da -= 0.95 * transforms[0].da;

			transforms[1].da *= 0.98;
			kSwitch *= 0.95;
		}

	}
	else if (roi.x + roi.width + (int)transforms[1].dx >= cols)
	{
		transforms[1].dx = (double)(cols - roi.x - roi.width);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[1].dx -= 0.95 * transforms[0].dx;
			//transforms[1].dy -= 0.95 * transforms[0].dy;
			//transforms[1].da -= 0.95 * transforms[0].da;
			transforms[1].da *= 0.98;
			kSwitch *= 0.95;
		}
	}

	if (roi.y + (int)transforms[1].dy < 0)
	{
		transforms[1].dy = (double)(1 - roi.y);
		if (tau_stab > 10) {
			tau_stab *= 0.9;
			//transforms[1].dx -= 0.95 * transforms[0].dx;
			//transforms[1].dy -= 0.95 * transforms[0].dy;
			//transforms[1].da -= 0.95 * transforms[0].da;
			transforms[1].da *= 0.98;
			kSwitch *= 0.95;
		}
	}
	else if (roi.y + roi.height + (int)transforms[1].dy >= rows)
	{
		transforms[1].dy = (double)(rows - roi.y - roi.height);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[1].dx -= 0.95 * transforms[0].dx;
			//transforms[1].dy -= 0.95 * transforms[0].dy;
			//transforms[1].da -= 0.95 * transforms[0].da;
			transforms[1].da *= 0.98;
			kSwitch *= 0.95;
		}
	}
	if (kSwitch < 1.0)
		tau_stab *= (4.0 + kSwitch) / 5.0;


	transforms[2].dx = (1.0 - 0.01) * transforms[2].dx + 0.01 * abs(transforms[1].dx);
	transforms[2].dy = (1.0 - 0.01) * transforms[2].dy + 0.01 * abs(transforms[1].dy);
	transforms[2].da = (1.0 - 0.01) * transforms[2].da + 0.01 * abs(transforms[1].da);

	//transforms[2].dx = 
	//KF.correct(transforms[1].dx);

	// Применение фильтра Калмана
	//cv::Mat prediction = KF.predict();


	if ((abs(transforms[0].dx) - 10.0 < 2.2 * transforms[3].dx || abs(transforms[0].dx) < 0.0) && (abs(transforms[0].dy) - 10.0 < 2.2 * transforms[3].dy || abs(transforms[0].dy) < 0.0) && (abs(transforms[0].da) - 0.01 < 2.2 * transforms[3].da || abs(transforms[0].da) < 0.0))
	{
		transforms[3].dx = (1.0 - 0.7) * transforms[3].dx + 0.7 * abs(transforms[0].dx);
		transforms[3].dy = (1.0 - 0.7) * transforms[3].dy + 0.7 * abs(transforms[0].dy);
		transforms[3].da = (1.0 - 0.7) * transforms[3].da + 0.7 * abs(transforms[0].da);
	}
	//transforms[3].dx = (1.0 - 0.7) * transforms[3].dx + 0.7 * abs(transforms[0].dx);
	//if (abs(transforms[0].dy) - 10.0 < 2.2 * transforms[3].dy || abs(transforms[0].dy) < 0.0)
	//transforms[3].dy = (1.0 - 0.7) * transforms[3].dy + 0.7 * abs(transforms[0].dy);
	//if (abs(transforms[0].da) - 0.01 < 2.2 * transforms[3].da || abs(transforms[0].da) < 0.0)
	//transforms[3].da = (1.0 - 0.7) * transforms[3].da + 0.7 * abs(transforms[0].da);
}


void kalmanFilter(vector<TransformParam>& movement, vector<TransformParam>& estimatedMovement)
{

}
