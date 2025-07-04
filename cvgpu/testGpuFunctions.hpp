#pragma once
//�������� ������������ ����� �� ������ ���������� Lucas-Kanade Optical Flow
//

#include <iostream> 
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp> 

#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>


#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include <thread>
#include <mutex>

#include "opencv2/video/tracking.hpp"

#include <opencv2/dnn.hpp>
// 
//#include <opencv2/cudev/ptr2d/warping.hpp>
using namespace cv;
using namespace std;


static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec);
static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec);

struct TransformParam
{
	TransformParam() {}
	TransformParam(double _dx, double _dy, double _da)
	{
		dx = _dx;
		dy = _dy;
		da = _da;
	}

	double dx;
	double dy;
	double da; // angle

	const void getTransform(Mat& T, Mat& frame, double crop)
	{
		double a = frame.cols;
		double b = frame.rows;
		double c = sqrt(a * a + b * b);

		// Reconstruct transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);
		//T.at<double>(0, 2) = dx + (a - c * cos(da + atan(b / a)))/2;
		//T.at<double>(1, 2) = dy + (b - c * sin(da + atan(b / a)))/2;
		T.at<double>(0, 2) = dx - a* crop*(1 - cos(atan(b / a) + da))/2;
		T.at<double>(1, 2) = dy - b* crop*(sin(atan(b / a) + da))/2;

	}

	const void getTransform(Mat& T, double a, double b, double c, double crop)
	{
		// Reconstruct transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);
		//T.at<double>(0, 2) = dx + (a - c * cos(da + atan(b / a)))/2;
		//T.at<double>(1, 2) = dy + (b - c * sin(da + atan(b / a)))/2;
		T.at<double>(0, 2) = dx - a * crop * (1 - cos(atan(b / a) + da)) / 2;
		T.at<double>(1, 2) = dy - b * crop * (sin(atan(b / a) + da)) / 2;
	}

	const void getTransform(Mat& T, double a, double b, double c, double atan_ba, double crop)
	{
		// Reconstruct transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);
		//T.at<double>(0, 2) = dx + (a - c * cos(da + atan_ba))/2;
		//T.at<double>(1, 2) = dy + (b - c * sin(da + atan_ba))/2;
		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;

	}
};

void iir(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, Mat& frame);

void fixBorder(Mat& frame_stabilized, double frame_part);


//void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray, cuda::GpuMat& gP0, vector<Point2f>& p0, double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms);
void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, vector <TransformParam>& transforms, Mat& T);

void calcPSF(Mat& outputImg, Size filterSize, int len, double theta);
void calcPSF(Mat& outputImg, Size filterSize, int len, double theta, Mat& temp);
void calcPSF_circle(Mat& outputImg, Size filterSize, int len, double theta);
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma = 5.0, double beta = 0.2);

void GcalcPSF(cuda::GpuMat& outputImg, Size filterSize, Size filterSizeCpu, int len, double theta);
//void GcalcPSF_circle(cuda::GpuMat& outputImg, Size filterSize, int len, double theta);
void GcalcPSFCircle(cuda::GpuMat& outputImg, Size filterSize, int len, double theta);
void Gfftshift(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg);
void Gfilter2DFreq(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& H);
void GcalcWnrFilter(const cuda::GpuMat& input_h_PSF, cuda::GpuMat& output_G, double nsr);

void Gedgetaper(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, double gamma = 5.0, double beta = 0.2);

void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(0, cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];


	Mat outputImg_norm;
	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
	cv::imshow("PSF", outputImg_norm);

}
void calcPSF(Mat& outputImg, Size filterSize, int len, double theta, Mat& temp)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(0, cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];


	//Mat outputImg_norm;
	normalize(outputImg, temp, 0, 255, NORM_MINMAX);
	//cv::imshow("PSF", outputImg_norm);

}
void calcPSF_circle(Mat& outputImg, Size filterSize, int len, double theta)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(cvRound(double(len) / 2.0), cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];


	Mat outputImg_norm;
	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
	cv::imshow("PSF", outputImg_norm);
}


void fftshift(const Mat& inputImg, Mat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));
	Mat q1(outputImg, Rect(cx, 0, cx, cy));
	Mat q2(outputImg, Rect(0, cy, cx, cy));
	Mat q3(outputImg, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
	Mat planes[2] = { Mat_<double>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_SCALE);

	Mat planesH[2] = { Mat_<double>(H.clone()), Mat::zeros(H.size(), CV_32F) };
	Mat complexH;
	merge(planesH, 2, complexH);
	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0);

	idft(complexIH, complexIH);
	split(complexIH, planes);
	outputImg = planes[0];
}

void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
	Mat h_PSF_shifted;
	fftshift(input_h_PSF, h_PSF_shifted);
	Mat planes[2] = { Mat_<double>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	Mat denom;
	pow(abs(planes[0]), 2, denom);
	denom += nsr;
	divide(planes[0], denom, output_G);
}

void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
{
	int Nx = inputImg.cols;
	int Ny = inputImg.rows;
	Mat w1(1, Nx, CV_32F, Scalar(0));
	Mat w2(Ny, 1, CV_32F, Scalar(0));

	double* p1 = w1.ptr<double>(0);
	double* p2 = w2.ptr<double>(0);
	double dx = double(2.0 * CV_PI / Nx);
	double x = double(-CV_PI);
	for (int i = 0; i < Nx; i++)
	{
		p1[i] = double(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
		x += dx;
	}
	double dy = double(2.0 * CV_PI / Ny);
	double y = double(-CV_PI);
	for (int i = 0; i < Ny; i++)
	{
		p2[i] = double(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
		y += dy;
	}
	Mat w = w2 * w1;
	multiply(inputImg, w, outputImg);
}



//~~~~~~~~~~~~~~
void GcalcPSF(cuda::GpuMat& outputImg, Size filterSize, Size psfSize, double len, double theta)
{
	// ������� GpuMat ��� ���������� ��������
	int scale = 8;
	cuda::GpuMat h(filterSize, CV_32F, Scalar(0));
	Mat hCpu(psfSize, CV_32F, Scalar(0));
	Mat hCpuBig(Size(psfSize.width*scale, psfSize.height*scale), CV_32F, Scalar(0));
	// ����� �������
	Point center(psfSize.width *scale/ 2, psfSize.height* scale/ 2);

	// ������� �������
	Size axes(scale, cvRound(double(len*scale +scale) / 2.0f));
	Size axes2(scale, cvRound(double(len*scale +scale) / 4.0f));
	Size axes3(scale, cvRound(double(len*scale +scale) / 6.0f));
	// ���� �������� �������
	double angle = 90.0 - theta;

	// ������ ������ �� GpuMat

	ellipse(hCpuBig, center, axes, angle, 0, 360, Scalar(0.2), FILLED);
	
	ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.4), FILLED);
	ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.9), FILLED);
	resize(hCpuBig, hCpu, psfSize, INTER_LINEAR);
	if(hCpu.cols > h.cols/2)
		resize(hCpu, hCpu, Size(h.cols/2-1, hCpu.rows), INTER_LINEAR);
	if (hCpu.rows > h.rows/2)
		resize(hCpu, hCpu, Size(hCpu.cols, h.rows/2-1), INTER_LINEAR);
	// 
		// �������� ����� ��������� ����� � ������� ����
	imshow("PSF Cpu", hCpu);
	//hCpu(Rect(0, 0, psfSize.width, psfSize.height)).copyTo(h(Rect((filterSize.width - psfSize.width) / 2, (filterSize.height - psfSize.height) / 2, psfSize.width, psfSize.height)));
	hCpu(Rect(0, 0, hCpu.cols, hCpu.rows)).copyTo(h(Rect((filterSize.width - hCpu.cols) / 2, (filterSize.height - hCpu.rows) / 2, hCpu.cols, hCpu.rows)));
	
	//h.upload(hCpu);



	// ��������� ��� �������� GpuMat
	Scalar summa = cuda::sum(h);

	// ����� GpuMat �� �����
	cuda::divide(h, Scalar(summa[0]), outputImg);


}

void GcalcPSFCircle(cuda::GpuMat& outputImg, Size filterSize, double len, double theta)
{
	// ������� GpuMat ��� ���������� ��������
	cuda::GpuMat h(filterSize, CV_32F, Scalar(0));
	Mat hCpu(filterSize, CV_32F, Scalar(0));
	// ����� �������
	Point center(filterSize.width / 2, filterSize.height / 2);

	// ������� �������
	Size axes(cvRound(double(len) / 2.0), cvRound(double(len) / 2.0));
	Size axes2(0, cvRound(double(len) / 4.0f));
	// ���� �������� �������
	double angle = 90.0 - theta;

	// ������ ������ �� GpuMat

	ellipse(hCpu, center, axes, angle, 0, 360, Scalar(255), FILLED);
	//ellipse(hCpu, center, axes2, angle, 0, 360, Scalar(255), FILLED);
	blur(hCpu, hCpu, Size(5, 5));
	h.upload(hCpu);
	
	// ��������� ��� �������� GpuMat
	Scalar summa = cuda::sum(h);

	// ����� GpuMat �� �����
	cuda::divide(h, Scalar(summa[0]), outputImg);
}

void Gfftshift(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	cuda::GpuMat q0(outputImg, Rect(0, 0, cx, cy));
	cuda::GpuMat q1(outputImg, Rect(cx, 0, cx, cy));
	cuda::GpuMat q2(outputImg, Rect(0, cy, cx, cy));
	cuda::GpuMat q3(outputImg, Rect(cx, cy, cx, cy));
	cuda::GpuMat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}
//

void Gfilter2DFreq(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& H)
{
	// ��������� ������� �����������
	cuda::GpuMat inputClone;
	inputImg.copyTo(inputClone);

	// ������� GpuMat ��� ������ �����
	cuda::GpuMat zeroMat(inputImg.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� � ����������� �������
	vector<cuda::GpuMat> planes = { inputClone, zeroMat };
	cuda::GpuMat complexInput;
	cuda::merge(planes, complexInput);

	// ������ �������������� �����
	//cuda::dft(complexInput, complexInput, complexInput.size(), DFT_SCALE | DFT_COMPLEX_OUTPUT);
	cuda::dft(complexInput, complexInput, complexInput.size(), DFT_SCALE);

	// ��������� ������
	cuda::GpuMat HClone;
	H.copyTo(HClone);

	// ������� GpuMat ��� ������ ����� �������
	cuda::GpuMat zeroMatH(H.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� ������� � ����������� �������
	vector<cuda::GpuMat> planesH = { HClone, zeroMatH };
	cuda::GpuMat complexH;
	cuda::merge(planesH, complexH);

	// ��������� ��������
	cuda::GpuMat complexOutput;
	cuda::mulSpectrums(complexInput, complexH, complexOutput, 0);

	// �������� �������������� �����
	cuda::dft(complexOutput, complexOutput, complexOutput.size(), DFT_INVERSE);

	// ��������� ����������� ������� �� ��� �����
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexOutput, planesOut);

	// ������ ��������� �������� ����������� ����������
	outputImg = planesOut[0];
}

void Gfilter2DFreqV2(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& complexH, Ptr<cuda::DFT>& forwardDFT, Ptr<cuda::DFT>& inverseDFT)
{
	// ��������� ������� �����������
	//cuda::GpuMat inputClone;
	//inputImg.copyTo(inputClone);

	// ������� GpuMat ��� ������ �����
	cuda::GpuMat zeroMat(inputImg.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� � ����������� �������
	vector<cuda::GpuMat> planes = { inputImg, zeroMat };
	cuda::GpuMat complexInput;
	cuda::merge(planes, complexInput);

	// ������ �������������� �����

	forwardDFT->compute(complexInput, complexInput);
	// ��������� ��������
	cuda::GpuMat complexOutput;
	cuda::mulSpectrums(complexInput, complexH, complexOutput, 0);
	// �������� �������������� �����
	inverseDFT->compute(complexOutput, complexOutput);

	// ��������� ����������� ������� �� ��� �����
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexOutput, planesOut);

	// ������ ��������� �������� ����������� ����������
	outputImg = planesOut[0];
}


//
void GcalcWnrFilter(const cuda::GpuMat& input_h_PSF, cuda::GpuMat& output_G, double nsr)
{
	// ������� ����� �������� �����������
	//cuda::GpuMat h_PSF_clone;
	//input_h_PSF.copyTo(h_PSF_clone);

	// ��������� ����� �����
	cuda::GpuMat h_PSF_shifted;
	Gfftshift(input_h_PSF, h_PSF_shifted);

	// ������� GpuMat ��� ������ �����
	cuda::GpuMat zeroMat(h_PSF_shifted.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� � ����������� �������
	vector<cuda::GpuMat> planes = { h_PSF_shifted, zeroMat };
	cuda::GpuMat complexI;
	cuda::merge(planes, complexI);

	// ������ �������������� �����
	//cuda::dft(complexI, complexI, complexI.size(), DFT_COMPLEX_OUTPUT);
	cuda::dft(complexI, complexI, complexI.size());

	// ��������� ����������� ������� �� ��� �����
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexI, planesOut);

	// ��������� �����������
	cuda::GpuMat denom;
	cuda::magnitude(planesOut[0], planesOut[1], denom);
	cuda::pow(denom, 2, denom);
	//denom += nsr;
	cuda::add(denom, nsr, denom);

	// �������
	cuda::divide(planesOut[0], denom, output_G);
}
//
//void Gedgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
//{
//	int Nx = inputImg.cols;
//	int Ny = inputImg.rows;
//	Mat w1(1, Nx, CV_32F, Scalar(0));
//	Mat w2(Ny, 1, CV_32F, Scalar(0));
//
//	double* p1 = w1.ptr<double>(0);
//	double* p2 = w2.ptr<double>(0);
//	double dx = double(2.0 * CV_PI / Nx);
//	double x = double(-CV_PI);
//	for (int i = 0; i < Nx; i++)
//	{
//		p1[i] = double(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
//		x += dx;
//	}
//	double dy = double(2.0 * CV_PI / Ny);
//	double y = double(-CV_PI);
//	for (int i = 0; i < Ny; i++)
//	{
//		p2[i] = double(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
//		y += dy;
//	}
//	Mat w = w2 * w1;
//	multiply(inputImg, w, outputImg);
//}

//~~~~~~~~~~~~~~




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

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, int cols, int rows, double& kSwitch)//, cv::KalmanFilter& KF)
{
	if ((abs(transforms[0].dx) - 10.0 < 1.2 * transforms[3].dx) && (abs(transforms[0].dy) - 10.0 < 1.2 * transforms[3].dy) && (abs(transforms[0].da) - 0.02 < 1.2 * transforms[3].da))//�������� �� ������ � ��������� ����������� ��������� ����������
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

		// ���������� ������� �������
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

static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}



void fixBorder(Mat& frame_stabilized, double frame_part)
{
	Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols / 2, frame_stabilized.rows / 2), 0, frame_part / (frame_part - 1));
	//Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols / 2, frame_stabilized.rows/2), 0, 0.5);
	warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
}

void GfixBorder(cuda::GpuMat& frame_stabilized, Rect& roi)
{
	//Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols / 2, frame_stabilized.rows / 2), 0, frame_part / (frame_part - 1));
	//Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols / 2, frame_stabilized.rows/2), 0, 0.5);
	//cuda::warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
	frame_stabilized = frame_stabilized(roi);
}


void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray, cuda::GpuMat& gP0, vector<Point2f>& p0, 
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms, 
	int& n, double& kSwitch, int a, int b, cuda::GpuMat& mask_device, bool& stab_possible)
{
	//for (int i = 0; i < 1; ++i)
	//{
		capture >> oldFrame;
		//resize(oldFrame, oldFrame, Size(oldFrame.cols * 3, oldFrame.rows * 3), 0.0, 0.0, INTER_LINEAR);
		gOldFrame.upload(oldFrame);
		cuda::resize(gOldFrame, gOldFrame, Size(a, b), 0.0, 0.0, cv::INTER_AREA);
		//cuda::resize(gOldFrame, gOldFrame, Size(gOldFrame.cols, gOldFrame.rows));

		cuda::bilateralFilter(gOldFrame, gOldFrame, 3, 3.0, 3.0); //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		//cvtColor(oldFrame, oldGray, COLOR_BGR2GRAY);
		cuda::cvtColor(gOldFrame, gOldGray, COLOR_BGR2GRAY);
		cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols / n, gOldGray.rows / n), 0.0, 0.0, cv::INTER_AREA);
		//imshow("Esc for exit.", oldFrame);
		//waitKey(1);
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
		//cuda::multiply(gP0, 4.0, gP0);
		if ((gP0.cols > 20)) {
			//if (i > 0)
			//	std::cout << "Iterations  " << i << std::endl;
			p0.clear();
			gP0.download(p0);
			stab_possible = true; //true
			//break;
		}
		else {
			//cerr << "Unable to find corners!" << endl;
			stab_possible = false;
		}

	//}

}


void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	int& n, double& kSwitch, int a, int b, cuda::GpuMat& mask_device, bool& stab_possible)
{
	//for (int i = 0; i < 1; ++i)
	//{
	
	//resize(oldFrame, oldFrame, Size(oldFrame.cols * 3, oldFrame.rows * 3), 0.0, 0.0, INTER_LINEAR);
	gOldFrame.upload(oldFrame);
	cuda::resize(gOldFrame, gOldFrame, Size(a, b), 0.0, 0.0, cv::INTER_AREA);
	//cuda::resize(gOldFrame, gOldFrame, Size(gOldFrame.cols, gOldFrame.rows));

	cuda::bilateralFilter(gOldFrame, gOldFrame, 3, 3.0, 3.0); //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	//cvtColor(oldFrame, oldGray, COLOR_BGR2GRAY);
	cuda::cvtColor(gOldFrame, gOldGray, COLOR_BGR2GRAY);
	cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols / n, gOldGray.rows / n), 0.0, 0.0, cv::INTER_AREA);
	// 
	//if (qualityLevel > 0.001 && harrisK > 0.001)
	//{
	//	qualityLevel *= 0.6;
	//	harrisK *= 0.6;
	//}
	//else
	//{
	//	if (maxCorners > 50)
	//	{
	//		maxCorners *= 0.98;
	//		d_features->setMaxCorners(maxCorners);
	//	}
	//}
	//for (int i = 0; i < 1;i++)
	//{
	//	transforms[i].dx *= kSwitch;
	//	transforms[i].dy *= kSwitch;
	//	transforms[i].da *= kSwitch;
	//}

	////d_features->detect(gOldGray, gP0, mask_device);
	////cuda::multiply(gP0, 4.0, gP0);
	//if ((gP0.cols > 20)) {
	//	//if (i > 0)
	//	//	std::cout << "Iterations  " << i << std::endl;
	//	p0.clear();
	//	gP0.download(p0);
	//	stab_possible = true;
	//	//break;
	//}
	//else {
	//	//cerr << "Unable to find corners!" << endl;
	//	stab_possible = false;
	//}

	//}
	stab_possible = false;
}

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, vector <TransformParam>& transforms, Mat& T, int n, Mat& T3d)
{
	for (uint i = 0; i < p1.size(); i++)
	{
		if (i == 0)
		{
			d = p1[0] - p0[0];
		}
		d = d + (p1[i] - p0[i]);
	}

	d = d / (int)p0.size();

	if (p0.empty() || p1.empty() || (p1.size() != p0.size()))
	{
		transforms[0] = TransformParam(-d.x * n, -d.y * n, 0.0);
		cout << "bull shit" << endl;
	}
	else
	{
		T = estimateAffine2D(p0, p1);
		//T = estimateAffinePartial2D(p0, p1, noArray());
		//transforms[0] = TransformParam(-(1*d.x + 1*T.at<double>(0, 2)) * n / 2, -(d.y*1 + 1*T.at<double>(1, 2)) * n / 2, ( 0.0 - 3.0*atan2(T.at<double>(1, 0), T.at<double>(0, 0)) + 1.0*transforms[0].da) / 4);
		//transforms[0] = TransformParam(-d.x, -d.y, 0.0);
		transforms[0] = TransformParam(-T.at<double>(0, 2), -T.at<double>(1, 2), -atan2(T.at<double>(1, 0), T.at<double>(0, 0)));
		
		//T3d = estimateAffine3D(p0, p1);
	}

	
}


void readFrameFromCapture(VideoCapture* capture, Mat* frame)
{
	*capture >> *frame;

}


void channelWiener(const cuda::GpuMat* gChannel, cuda::GpuMat* gChannelWiener,
	const cuda::GpuMat* complexH, cv::Ptr<cuda::DFT>* forwardDFT, cv::Ptr<cuda::DFT>* inverseDFT)
{

	Gfilter2DFreqV2(*gChannel, *gChannelWiener, *complexH, *forwardDFT, *inverseDFT);
}
