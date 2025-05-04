#pragma once
//Алгоритм стабилизации видео на основе вычисление Lucas-Kanade Optical Flow
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


#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>
#include <cmath>


#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
		T.at<double>(0, 2) = dx - a * crop * (1 - cos(atan(b / a) + da)) / 2;
		T.at<double>(1, 2) = dy - b * crop * (sin(atan(b / a) + da)) / 2;

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
//void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, vector <TransformParam>& transforms, Mat& T);

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



//~~~~~~~~~~~~~~~~~~~~~~~~~~~
void GcalcPSF(cuda::GpuMat& outputImg, Size filterSize, Size psfSize, double len, double theta)
{
	// Создаем GpuMat для временного хранения
	int scale = 8;
	cuda::GpuMat h(filterSize, CV_32F, Scalar(0));
	Mat hCpu(psfSize, CV_32F, Scalar(0));
	Mat hCpuBig(Size(psfSize.width * scale, psfSize.height * scale), CV_32F, Scalar(0));
	// Центр эллипса
	Point center(psfSize.width * scale / 2, psfSize.height * scale / 2);

	// Радиусы эллипса
	Size axes(scale, cvRound(double(len * scale + scale) / 2.0f));
	Size axes2(scale, cvRound(double(len * scale + scale) / 4.0f));
	Size axes3(scale, cvRound(double(len * scale + scale) / 6.0f));
	// Углы поворота эллипса
	double angle = 90.0 - theta;

	// Рисуем эллипс на GpuMat

	ellipse(hCpuBig, center, axes, angle, 0, 360, Scalar(0.2), FILLED);

	ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.4), FILLED);
	ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.9), FILLED);
	resize(hCpuBig, hCpu, psfSize, INTER_LINEAR);
	if (hCpu.cols > h.cols / 2)
		resize(hCpu, hCpu, Size(h.cols / 2 - 1, hCpu.rows), INTER_LINEAR);
	if (hCpu.rows > h.rows / 2)
		resize(hCpu, hCpu, Size(hCpu.cols, h.rows / 2 - 1), INTER_LINEAR);
	// 
		// Копируем часть исходного кадра в большой кадр
	imshow("PSF Cpu", hCpu);
	//hCpu(Rect(0, 0, psfSize.width, psfSize.height)).copyTo(h(Rect((filterSize.width - psfSize.width) / 2, (filterSize.height - psfSize.height) / 2, psfSize.width, psfSize.height)));
	hCpu(Rect(0, 0, hCpu.cols, hCpu.rows)).copyTo(h(Rect((filterSize.width - hCpu.cols) / 2, (filterSize.height - hCpu.rows) / 2, hCpu.cols, hCpu.rows)));

	//h.upload(hCpu);



	// Суммируем все элементы GpuMat
	Scalar summa = cuda::sum(h);

	// Делим GpuMat на сумму
	cuda::divide(h, Scalar(summa[0]), outputImg);


}

void GcalcPSFCircle(cuda::GpuMat& outputImg, Size filterSize, double len, double theta)
{
	// Создаем GpuMat для временного хранения
	cuda::GpuMat h(filterSize, CV_32F, Scalar(0));
	Mat hCpu(filterSize, CV_32F, Scalar(0));
	// Центр эллипса
	Point center(filterSize.width / 2, filterSize.height / 2);

	// Радиусы эллипса
	Size axes(cvRound(double(len) / 2.0), cvRound(double(len) / 2.0));
	Size axes2(0, cvRound(double(len) / 4.0f));
	// Углы поворота эллипса
	double angle = 90.0 - theta;

	// Рисуем эллипс на GpuMat

	ellipse(hCpu, center, axes, angle, 0, 360, Scalar(255), FILLED);
	//ellipse(hCpu, center, axes2, angle, 0, 360, Scalar(255), FILLED);
	blur(hCpu, hCpu, Size(5, 5));
	h.upload(hCpu);

	// Суммируем все элементы GpuMat
	Scalar summa = cuda::sum(h);

	// Делим GpuMat на сумму
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
	// Клонируем входное изображение
	cuda::GpuMat inputClone;
	inputImg.copyTo(inputClone);

	// Создаем GpuMat для мнимой части
	cuda::GpuMat zeroMat(inputImg.size(), CV_32F, Scalar(0));

	// Объединяем действительную и мнимую часть в комплексную матрицу
	vector<cuda::GpuMat> planes = { inputClone, zeroMat };
	cuda::GpuMat complexInput;
	cuda::merge(planes, complexInput);

	// Прямое преобразование Фурье
	//cuda::dft(complexInput, complexInput, complexInput.size(), DFT_SCALE | DFT_COMPLEX_OUTPUT);
	cuda::dft(complexInput, complexInput, complexInput.size(), DFT_SCALE);

	// Клонируем фильтр
	cuda::GpuMat HClone;
	H.copyTo(HClone);

	// Создаем GpuMat для мнимой части фильтра
	cuda::GpuMat zeroMatH(H.size(), CV_32F, Scalar(0));

	// Объединяем действительную и мнимую часть фильтра в комплексную матрицу
	vector<cuda::GpuMat> planesH = { HClone, zeroMatH };
	cuda::GpuMat complexH;
	cuda::merge(planesH, complexH);

	// Умножение спектров
	cuda::GpuMat complexOutput;
	cuda::mulSpectrums(complexInput, complexH, complexOutput, 0);

	// Обратное преобразование Фурье
	cuda::dft(complexOutput, complexOutput, complexOutput.size(), DFT_INVERSE);

	// Разделяем комплексную матрицу на две части
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexOutput, planesOut);

	// Первый компонент является результатом фильтрации
	outputImg = planesOut[0];
}

void Gfilter2DFreqV2(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& complexH, Ptr<cuda::DFT>& forwardDFT, Ptr<cuda::DFT>& inverseDFT)
{
	// Клонируем входное изображение
	//cuda::GpuMat inputClone;
	//inputImg.copyTo(inputClone);

	// Создаем GpuMat для мнимой части
	cuda::GpuMat zeroMat(inputImg.size(), CV_32F, Scalar(0));

	// Объединяем действительную и мнимую часть в комплексную матрицу
	vector<cuda::GpuMat> planes = { inputImg, zeroMat };
	cuda::GpuMat complexInput;
	cuda::merge(planes, complexInput);

	// Прямое преобразование Фурье

	forwardDFT->compute(complexInput, complexInput);
	// Умножение спектров
	cuda::GpuMat complexOutput;
	cuda::mulSpectrums(complexInput, complexH, complexOutput, 0);
	// Обратное преобразование Фурье
	inverseDFT->compute(complexOutput, complexOutput);

	// Разделяем комплексную матрицу на две части
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexOutput, planesOut);

	// Первый компонент является результатом фильтрации
	outputImg = planesOut[0];
}


//
void GcalcWnrFilter(const cuda::GpuMat& input_h_PSF, cuda::GpuMat& output_G, double nsr)
{
	// Создаем копию входного изображения
	//cuda::GpuMat h_PSF_clone;
	//input_h_PSF.copyTo(h_PSF_clone);

	// Применяем сдвиг Фурье
	cuda::GpuMat h_PSF_shifted;
	Gfftshift(input_h_PSF, h_PSF_shifted);

	// Создаем GpuMat для мнимой части
	cuda::GpuMat zeroMat(h_PSF_shifted.size(), CV_32F, Scalar(0));

	// Объединяем действительную и мнимую часть в комплексную матрицу
	vector<cuda::GpuMat> planes = { h_PSF_shifted, zeroMat };
	cuda::GpuMat complexI;
	cuda::merge(planes, complexI);

	// Прямое преобразование Фурье
	//cuda::dft(complexI, complexI, complexI.size(), DFT_COMPLEX_OUTPUT);
	cuda::dft(complexI, complexI, complexI.size());

	// Разделяем комплексную матрицу на две части
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexI, planesOut);

	// Вычисляем знаменатель
	cuda::GpuMat denom;
	cuda::magnitude(planesOut[0], planesOut[1], denom);
	cuda::pow(denom, 2, denom);
	//denom += nsr;
	cuda::add(denom, nsr, denom);

	// Деление
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~




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

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, int cols, int rows, double& kSwitch, vector<TransformParam>& velocity)//, cv::KalmanFilter& KF)
{
	//if ((abs(transforms[0].dx) - 10.0 < 1.2 * transforms[3].dx) && (abs(transforms[0].dy) - 10.0 < 1.2 * transforms[3].dy) && (abs(transforms[0].da) - 0.02 < 1.2 * transforms[3].da))//проверка на выброс и предельно минимальную амплитуду отклонения
	if ((abs(transforms[0].dx) - 10.0 < 3.0 * transforms[3].dx) && (abs(transforms[0].dy) - 10.0 < 3.0 * transforms[3].dy) && (abs(transforms[0].da) - 0.02 < 3.0 * transforms[3].da))//проверка на выброс и предельно минимальную амплитуду отклонения
	{
		transforms[1].dx = kSwitch * (transforms[1].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[0].dx); //накопление по перемещнию внутри кадра
		transforms[1].dy = kSwitch * (transforms[1].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[0].dy);
		transforms[1].da = kSwitch * (transforms[1].da * (1.7 * tau_stab - 1.0) / (1.7 * tau_stab) + kSwitch * transforms[0].da);
	}

	if (transforms[1].da > 2.0)
		transforms[1].da = 2.0;

	if (transforms[1].da < -2.0)
		transforms[1].da = -2.0;

	if (tau_stab < 30.0) 
		tau_stab *= 1.1;
	
	if (tau_stab < 50.0 && !(abs(transforms[1].dx) > cols / 2 || abs(transforms[1].dy) > rows / 2)) 
		tau_stab *= 1.1;
	
	if (tau_stab < 120.0 && !(abs(transforms[1].dx) > cols / 3 || abs(transforms[1].dy) > rows / 3)) 
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
			transforms[1].da *= 0.99;
			kSwitch *= 0.95;
		}

	}
	else if (roi.x + roi.width + (int)transforms[1].dx >= cols)
	{
		transforms[1].dx = (double)(cols - roi.x - roi.width);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[1].da *= 0.99;
			kSwitch *= 0.95;
		}
	}

	if (roi.y + (int)transforms[1].dy < 0)
	{
		transforms[1].dy = (double)(1 - roi.y);
		if (tau_stab > 10) {
			tau_stab *= 0.9;
			transforms[1].da *= 0.99;
			kSwitch *= 0.95;
		}
	}
	else if (roi.y + roi.height + (int)transforms[1].dy >= rows)
	{
		transforms[1].dy = (double)(rows - roi.y - roi.height);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[1].da *= 0.99;
			kSwitch *= 0.95;
		}
	}

	if (kSwitch < 1.0)
		tau_stab *= (4.0 + kSwitch) / 5.0;
	
	transforms[2].dx = (1.0 - 0.01) * transforms[2].dx + 0.01 * abs(transforms[1].dx);
	transforms[2].dy = (1.0 - 0.01) * transforms[2].dy + 0.01 * abs(transforms[1].dy);
	transforms[2].da = (1.0 - 0.01) * transforms[2].da + 0.01 * abs(transforms[1].da);

	//if ((abs(transforms[0].dx) - 10.0 < 2.2 * transforms[3].dx || abs(transforms[0].dx) < 0.0) && (abs(transforms[0].dy) - 10.0 < 2.2 * transforms[3].dy || abs(transforms[0].dy) < 0.0) && (abs(transforms[0].da) - 0.01 < 2.2 * transforms[3].da || abs(transforms[0].da) < 0.0))
	if (true)
	{
		transforms[3].dx = (1.0 - 0.9) * transforms[3].dx + 0.9 * abs(transforms[0].dx); //среднее абсолютное отклонение колебаний
		transforms[3].dy = (1.0 - 0.9) * transforms[3].dy + 0.9 * abs(transforms[0].dy);
		transforms[3].da = (1.0 - 0.9) * transforms[3].da + 0.9 * abs(transforms[0].da);
	}


	velocity[0].dx = velocity[0].dx * 63 / 64 + transforms[0].dx / 64;
	velocity[0].dy = velocity[0].dy * 63 / 64 + transforms[0].dy / 64;
	velocity[0].da = velocity[0].da * 63 / 64 + transforms[0].da / 64;
}

void iirAdaptiveHighPass(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, int cols, int rows, double& kSwitch)//, cv::KalmanFilter& KF)
{
	if ((abs(transforms[0].dx) - (double)rows/2 < 1.2 * transforms[3].dx) && (abs(transforms[0].dy) - (double)rows / 2 < 1.2 * transforms[3].dy) && (abs(transforms[0].da) - 0.2 < 3.0 * transforms[3].da))//проверка на выброс и предельно минимальную амплитуду отклонения
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
	capture >> oldFrame;

	gOldFrame.upload(oldFrame);

	cuda::resize(gOldFrame, gOldFrame, Size(a, b), 0.0, 0.0, cv::INTER_AREA);
	cuda::bilateralFilter(gOldFrame, gOldFrame, 3, 3.0, 3.0); //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cuda::cvtColor(gOldFrame, gOldGray, COLOR_BGR2GRAY);
	cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols / n, gOldGray.rows / n), 0.0, 0.0, cv::INTER_AREA);

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

	//}

}


void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	int& n, double& kSwitch, int a, int b, cuda::GpuMat& mask_device, bool& stab_possible)
{
	gOldFrame.upload(oldFrame);
	cuda::resize(gOldFrame, gOldFrame, Size(a, b), 0.0, 0.0, cv::INTER_AREA);

	cuda::bilateralFilter(gOldFrame, gOldFrame, 3, 3.0, 3.0);
	cuda::cvtColor(gOldFrame, gOldGray, COLOR_BGR2GRAY);
	cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols / n, gOldGray.rows / n), 0.0, 0.0, cv::INTER_AREA);
	stab_possible = false;
}

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, vector <TransformParam>& transforms, Mat& T, int n)
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

		//transforms[0] = TransformParam(-(T.at<double>(0, 2) + 3*d.x)/4, -(T.at<double>(1, 2) + 3*d.y)/4, -atan2(T.at<double>(1, 0), T.at<double>(0, 0)));
		transforms[0] = TransformParam(-T.at<double>(0, 2), -T.at<double>(1, 2), -atan2(T.at<double>(1, 0), T.at<double>(0, 0)));

	}


}


Mat videoStabHomograpy(cuda::GpuMat& gFrame, vector<Point2f>& p0, vector<Point2f>& p1, vector<Mat>& homoTransforms)
{
	// Вычисляем гомографию
	Mat H;
	if (p0.size() >= 4) {
		H = findHomography(p0, p1, RANSAC);
		homoTransforms.push_back(H);
	}
	else {
		homoTransforms.push_back(Mat::eye(3, 3, CV_64F));
	}

	if (homoTransforms.size() < 400) {
		return homoTransforms.back();
	}
	// Вычисляем среднее преобразование в окне
	Mat sumH = Mat::zeros(3, 3, CV_64F);
	int count = 0;
	for (int i = homoTransforms.size() - 400; i < homoTransforms.size(); i++) {
		sumH += homoTransforms[i];
		count++;
	}
	return sumH / count;
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


// Константы
const double DEG_TO_RAD = CV_PI / 180.0;
const double RAD_TO_DEG = 180.0 / CV_PI;

// Структура для хранения углов ориентации
struct OrientationAngles {
	double roll;    // Крен (радианы)
	double pitch;   // Тангаж (радианы)
	double yaw;     // Рыскание (радианы)
};

// Функция для оценки ориентации по гомографии
OrientationAngles estimateOrientationFromHomography(const cv::Mat& H, double focalLength, double cx, double cy) {
	OrientationAngles angles;

	// Нормализация матрицы гомографии
	cv::Mat Hnorm = H / H.at<double>(2, 2);

	// Извлечение компонент вращения
	double h11 = Hnorm.at<double>(0, 0), h12 = Hnorm.at<double>(0, 1), h13 = Hnorm.at<double>(0, 2);
	double h21 = Hnorm.at<double>(1, 0), h22 = Hnorm.at<double>(1, 1), h23 = Hnorm.at<double>(1, 2);
	double h31 = Hnorm.at<double>(2, 0), h32 = Hnorm.at<double>(2, 1), h33 = Hnorm.at<double>(2, 2);

	// Вычисление углов
	angles.yaw = atan2(h21, h11);
	angles.pitch = atan2(-h31, sqrt(h32 * h32 + h33 * h33));
	angles.roll = atan2(h32, h33);

	return angles;
}

// Основная функция обработки
OrientationAngles estimateUAVOrientation(cv::cuda::GpuMat& currentFrame, cv::cuda::GpuMat& previousFrame,
	double focalLength, double cx, double cy) {
	// Преобразование в grayscale
	cv::cuda::GpuMat prevGray, currGray;
	cv::cuda::cvtColor(previousFrame, prevGray, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(currentFrame, currGray, cv::COLOR_BGR2GRAY);

	// Детекция особенностей (используем ORB на GPU)
	auto detector = cv::cuda::ORB::create(1000);

	// Для хранения ключевых точек и дескрипторов
	std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
	cv::cuda::GpuMat prevDescriptorsGPU, currDescriptorsGPU;

	// Детекция и вычисление дескрипторов
	detector->detectAndCompute(prevGray, cv::cuda::GpuMat(), prevKeypoints, prevDescriptorsGPU);
	detector->detectAndCompute(currGray, cv::cuda::GpuMat(), currKeypoints, currDescriptorsGPU);

	// Конвертируем дескрипторы в CPU формат
	cv::Mat prevDescriptors, currDescriptors;
	prevDescriptorsGPU.download(prevDescriptors);
	currDescriptorsGPU.download(currDescriptors);

	// Сопоставление особенностей
	auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<cv::DMatch> matches;
	matcher->match(prevDescriptors, currDescriptors, matches);

	// Фильтрация хороших соответствий
	double minDist = DBL_MAX;
	for (const auto& m : matches) {
		if (m.distance < minDist) minDist = m.distance;
	}

	std::vector<cv::DMatch> goodMatches;
	for (const auto& m : matches) {
		if (m.distance < std::max(2.0 * minDist, 30.0)) {
			goodMatches.push_back(m);
		}
	}

	// Получаем точки соответствий
	std::vector<cv::Point2f> prevPoints, currPoints;
	for (const auto& m : goodMatches) {
		prevPoints.push_back(prevKeypoints[m.queryIdx].pt);
		currPoints.push_back(currKeypoints[m.trainIdx].pt);
	}

	// Вычисляем гомографию
	cv::Mat H;
	if (prevPoints.size() >= 4) {
		H = cv::findHomography(prevPoints, currPoints, cv::RANSAC, 3.0);
	}
	else {
		// Недостаточно точек для оценки
		return OrientationAngles{ 0, 0, 0 };
	}

	// Оцениваем углы ориентации из гомографии
	return estimateOrientationFromHomography(H, focalLength, cx, cy);
}


int camera_calibration(int argc, char** argv) {

	(void)argc;
	(void)argv;

	std::vector<cv::String> fileNames;
	cv::glob("D:/CV_camera_calibration_images/*.jpg", fileNames, false);
	//cv::glob("D:/CV_camera_calibration_images/png/Image*.png", fileNames, false);
	//cv::glob("../calibration/Image*.png", fileNames, false);
	cv::Size patternSize(18 - 1, 12 - 1); //18 12
	std::vector<std::vector<cv::Point2f>> q(fileNames.size());

	std::vector<std::vector<cv::Point3f>> Q;
	// 1. Generate checkerboard (world) coordinates Q. The board has 25 x 18
	// fields with a size of 15x15mm

	int checkerBoard[2] = { 18,12 };
	// Defining the world coordinates for 3D points
	std::vector<cv::Point3f> objp;
	for (int i = 1; i < checkerBoard[1]; i++) {
		for (int j = 1; j < checkerBoard[0]; j++) {
			objp.push_back(cv::Point3f(j, i, 0));
		}
	}

	std::vector<cv::Point2f> imgPoint;
	// Detect feature points
	std::size_t i = 0;
	for (auto const& f : fileNames) {
		std::cout << std::string(f) << std::endl;

		// 2. Read in the image an call cv::findChessboardCorners()
		cv::Mat img = cv::imread(fileNames[i]);
		cv::Mat gray;

		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

		bool patternFound = cv::findChessboardCorners(gray, patternSize, q[i], cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

		// 2. Use cv::cornerSubPix() to refine the found corner detections
		if (patternFound) {
			cv::cornerSubPix(gray, q[i], cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
			Q.push_back(objp);
		}

		// Display
		cv::drawChessboardCorners(img, patternSize, q[i], patternFound);
		cv::imshow("chessboard detection", img);
		//cv::waitKey(0);

		i++;
	}


	cv::Matx33f K(cv::Matx33f::eye());  // intrinsic camera matrix
	cv::Vec<float, 5> k(0, 0, 0, 0, 0); // distortion coefficients

	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<double> stdIntrinsics, stdExtrinsics, perViewErrors;
	int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
		cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;
	cv::Size frameSize(1280, 720);

	std::cout << "Calibrating..." << std::endl;
	// 4. Call "float error = cv::calibrateCamera()" with the input coordinates
	// and output parameters as declared above...

	float error = cv::calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs, flags);

	std::cout << "Reprojection error = " << error << "\nK =\n"
		<< K << "\nk=\n"
		<< k << std::endl;

	// Precompute lens correction interpolation
	cv::Mat mapX, mapY;
	cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1,
		mapX, mapY);

	// Show lens corrected images
	for (auto const& f : fileNames) {
		std::cout << std::string(f) << std::endl;

		cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);

		cv::Mat imgUndistorted;
		// 5. Remap the image using the precomputed interpolation maps.
		cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);

		// Display
		cv::imshow("undistorted image", imgUndistorted);
		cv::waitKey(0);
	}

	cv::VideoCapture cap(0);
	cv::Mat frame, undistorted;
	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		// Исправление искажений
		cv::remap(frame, undistorted, mapX, mapY, cv::INTER_LINEAR);

		// Показ исходного и исправленного изображения
		cv::imshow("Original", frame);
		cv::imshow("Undistorted", undistorted);

		if (cv::waitKey(1) == 27) break; // ESC для выхода
	}


	return 0;
}

bool keyResponse(int& keyboard, Mat& frame, Mat& croppedImg, const double& a, const double& b, double& nsr, bool& wienner, bool& threadwiener, double& Q, double& tauStab, double& framePart, Rect& roi)
{
	if (keyboard == 'c')
	{
		imwrite("imgInCam.jpg", frame);
		imwrite("imgOutCam.jpg", croppedImg);
	}
	if (keyboard == 'q' || keyboard == 27)
		return true;
	if (keyboard == '8')
	{
		nsr = nsr * 0.8;
	}
	if (keyboard == '7')
	{
		nsr = nsr * 1.25;
	}
	if (keyboard == '1')
	{
		wienner = wienner ^ true;
	}
	if (keyboard == 't')
	{
		threadwiener = threadwiener ^ true;
	}

	if (keyboard == '6')
	{
		if (Q < 20.0)
			Q = Q * 1.05;
	}
	if (keyboard == '5')
	{
		if (Q > 1.0)
			Q = Q * 0.95;
		if (Q < 1.0)
			Q = 1.0;
	}
	if (keyboard == '4')
	{
		if (tauStab < 4000)
			tauStab = tauStab * 2;

	}
	if (keyboard == '3')
	{
		if (tauStab > 4)
			tauStab = tauStab / 2;
	}
	if (keyboard == 's' || keyboard == 'S')
	{
		if (framePart < 0.95)
		{
			framePart *= 1.01;
			if (framePart > 0.9)
				framePart = 0.9;
			roi.x = a * ((1.0 - framePart) / 2.0);
			roi.y = b * ((1.0 - framePart) / 2.0);
			roi.width = a * framePart;
			roi.height = b * framePart;

		}
	}
	if (keyboard == 'w' || keyboard == 'W')
	{
		if (framePart > 0.05)
		{
			framePart *= 0.99;
			if (framePart < 0.05)
				framePart = 0.05;
			roi.x = a * ((1.0 - framePart) / 2.0);
			roi.y = b * ((1.0 - framePart) / 2.0);
			roi.width = a * framePart;
			roi.height = b * framePart;

		}
	}
	return false;
}