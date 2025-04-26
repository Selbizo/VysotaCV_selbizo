//
////Алгоритм стабилизации видео на основе вычисление Lucas-Kanade Optical Flow
////
//#include <iostream>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/video.hpp>
//#include <opencv2/opencv.hpp>
//#include <cassert>
//#include <cmath>
//#include <fstream>
//
//using namespace cv;
//using namespace std;
//
//struct TransformParam
//{
//	TransformParam() {}
//	TransformParam(double _dx, double _dy, double _da)
//	{
//		dx = _dx;
//		dy = _dy;
//		da = _da;
//	}
//
//	double dx;
//	double dy;
//	double da; // angle
//
//	void getTransform(Mat& T, Mat& frame)
//	{
//		double a = frame.cols;
//		double b = frame.rows;
//		double c = sqrt(a * a + b * b);
//
//		// Reconstruct transformation matrix accordingly to new values
//		T.at<double>(0, 0) = cos(da);
//		T.at<double>(0, 1) = -sin(da);
//		T.at<double>(1, 0) = sin(da);
//		T.at<double>(1, 1) = cos(da);
//		T.at<double>(0, 2) = dx + (a / 2 - c / 2 * cos(da + atan(b / a)));
//		T.at<double>(1, 2) = dy + (b / 2 - c / 2 * sin(da + atan(b / a)));
//
//	}
//
//	void getTransform(Mat& T, double a, double b, double c)
//	{
//		// Reconstruct transformation matrix accordingly to new values
//		T.at<double>(0, 0) = cos(da);
//		T.at<double>(0, 1) = -sin(da);
//		T.at<double>(1, 0) = sin(da);
//		T.at<double>(1, 1) = cos(da);
//		T.at<double>(0, 2) = dx + (a / 2 - c / 2 * cos(da + atan(b / a)));
//		T.at<double>(1, 2) = dy + (b / 2 - c / 2 * sin(da + atan(b / a)));
//	}
//
//	void getTransform(Mat& T, double a, double b, double c, double atan_ba)
//	{
//		// Reconstruct transformation matrix accordingly to new values
//		T.at<double>(0, 0) = cos(da);
//		T.at<double>(0, 1) = -sin(da);
//		T.at<double>(1, 0) = sin(da);
//		T.at<double>(1, 1) = cos(da);
//		T.at<double>(0, 2) = dx + (a / 2 - c / 2 * cos(da + atan_ba));
//		T.at<double>(1, 2) = dy + (b / 2 - c / 2 * sin(da + atan_ba));
//
//	}
//};
//
//struct Trajectory
//{
//	//Trajectory() {}
//	Trajectory(double _x, double _y, double _a) {
//		x = _x;
//		y = _y;
//		a = _a;
//	}
//
//	double x;
//	double y;
//	double a; // angle
//};
//
//void iir(vector<TransformParam>& transforms, double tau_stab)
//{
//	transforms[1].dx = transforms[1].dx * (tau_stab - 1.0) / tau_stab + transforms[0].dx;
//	transforms[1].dy = transforms[1].dy * (tau_stab - 1.0) / tau_stab + transforms[0].dy;
//	transforms[1].da = transforms[1].da * (1.0 * tau_stab - 1.0) / (1.0 * tau_stab) + transforms[0].da;
//}
//
//void fixBorder(Mat& frame_stabilized, double frame_part)
//{
//	Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols / 2, frame_stabilized.rows / 2), 0, frame_part / (frame_part - 1));
//	//Mat T = getRotationMatrix2D(Point2f(frame_stabilized.cols / 2, frame_stabilized.rows/2), 0, 0.5);
//	warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
//}
//
//
//
//
//void calcPSF(Mat& outputImg, Size filterSize, int len, double theta);
//void calcPSF_circle(Mat& outputImg, Size filterSize, int len, double theta);
//void fftshift(const Mat& inputImg, Mat& outputImg);
//void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
//void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
//void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma = 5.0, double beta = 0.2);
//
//int main(int argc, char** argv)
//{
//	VideoCapture capture(0);
//	//VideoCapture capture("video1.mp4");
//	//VideoCapture capture("video2.mp4");
//
//	if (!capture.isOpened()) {
//		//error in opening the video input
//		cerr << "Unable to connect camera!" << endl;
//		return 0;
//	}
//
//	// Create some random colors
//	vector<Scalar> colors;
//	RNG rng;
//	for (int i = 0; i < 1000; i++)
//	{
//		int b = rng.uniform(206, 256);
//		int g = rng.uniform(124, 234);
//		int r = rng.uniform(0, 52);
//		colors.push_back(Scalar(b, g, r));
//	}
//	// переменные для поиска характерных точек
//	Mat old_frame, old_gray;
//	vector<Point2f> p0, p1;
//	Point2f d;
//	d = Point2f(0.0f, 0.0f);
//
//	vector<uchar> status;
//	vector<float> err;
//	TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 20, 0.01);
//
//	int max_corners = 100; //50
//	double quality_level = 0.04; //0.01
//	double min_distance = 1.0;
//	int block_size = 9;
//	double harris_quality = 0.03;
//	// Take first frame and find corners in it
//	capture >> old_frame; //707ms
//	//resize(old_frame, old_frame, Size(old_frame.cols/2, old_frame.rows/2), 0.0, 0.0, INTER_AREA);
//
//	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY); //3 ms
//	goodFeaturesToTrack(old_gray, p0, max_corners, quality_level, min_distance, Mat(), block_size, true, harris_quality);
//
//	double tau_stab = 50.0;
//
//	int frame_part = 4;
//	Rect roi;
//	roi.x = old_gray.cols / (frame_part * 2);
//	roi.y = old_gray.rows / (frame_part * 2);
//	roi.width = old_gray.cols * (frame_part - 1) / (frame_part);
//	roi.height = old_gray.rows * (frame_part - 1) / (frame_part);
//
//	const double a = old_frame.cols;
//	const double b = old_frame.rows;
//	const double c = sqrt(a * a + b * b);
//	const double atan_ba = atan2(b, a);
//
//	// переменные для фильтра Виннера
//	Mat Hw, h, frame_gray_wienner;
//	bool wienner = false;
//	double nsr = 0.02;
//	double Q = 4.0; // скважность считывания кадра на камере (выдержка к частоте кадров) (умножена на 10)
//	int LEN = 0;
//	double THETA = 0.0;
//
//	//для обработки трех каналов по Виннеру
//	vector<Mat> channels(3);
//	vector<Mat> channels_wienner(3);
//	Mat frame_wienner;
//
//	//Для отображения надписей на кадре
//	setlocale(LC_ALL, "RU");
//	vector <Point> textOrg(10);
//
//	for (int i = 0; i < 10; i++)
//	{
//		textOrg[i].x = 5;
//		textOrg[i].y = 20 * (i + 1);
//	}
//
//	int fontFace = FONT_HERSHEY_PLAIN;
//	double fontScale = 0.9;
//	Scalar color(5, 10, 230);
//
//	//для вывода изображения на дисплей
//	Mat cropped_img, frame_crop;
//	Mat frame, frame_gray;
//
//	Mat T;
//	Mat T_stab(2, 3, CV_64F);
//	Mat frame_stabilized, frame_out;
//	vector <TransformParam> transforms(2);
//	for (int i = 0; i < 2;i++)
//	{
//		transforms[i].dx = 0.0;
//		transforms[i].dy = 0.0;
//		transforms[i].da = 0.0;
//	}
//
//	//vector <TransformParam> transforms(SMOOTHING_RADIUS);
//
//	int frame_cnt = 0;
//	double seconds = 0.0;
//	clock_t start = clock();
//	clock_t end = clock();
//	while (true) {
//
//		capture >> frame;
//		//resize(frame, frame, Size(frame.cols / 2, frame.rows / 2), 0.0, 0.0, INTER_LINEAR);
//		frame_cnt++;
//		if (frame_cnt == 10)
//		{
//
//			end = clock();
//			seconds = (double)(end - start) / CLOCKS_PER_SEC / frame_cnt;
//			frame_cnt = 0;
//			start = clock();
//		}
//
//
//		if (frame.empty())
//		{
//			break;
//		}
//
//		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
//		//medianBlur(frame_gray, frame_gray, 5);
//
//		if (p0.size() < max_corners * 2 / 7)
//		{
//			while (true)
//			{
//				capture >> old_frame;
//				//resize(old_frame, old_frame, Size(old_frame.cols / 2, old_frame.rows / 2), 0.0, 0.0, INTER_LINEAR);
//				cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
//				quality_level *= 0.9;
//				harris_quality *= 0.9;
//				goodFeaturesToTrack(old_gray, p0, max_corners, quality_level, min_distance, Mat(), block_size, true, harris_quality);
//
//				if (p0.size() > max_corners * 1 / 7) {
//					break;
//				}
//			}
//			calcOpticalFlowPyrLK(frame_gray, frame_gray, p0, p1, status, err, Size(41, 41), 15, criteria);
//		}
//
//		if (p0.size() > max_corners * 5 / 6) {
//			if (quality_level < 1.0)
//			{
//				quality_level *= 1.1;
//				harris_quality *= 1.1;
//			}
//		}
//
//		calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(41, 41), 15, criteria);
//		vector<Point2f> good_new;
//
//		for (uint i = 0; i < p0.size(); i++)
//		{
//			// Select good points
//			if (status[i] == 1) {
//				good_new.push_back(p1[i]);
//			}
//		}
//
//		//stabilization part
//		for (uint i = 0; i < p1.size(); i++)
//		{
//			if (i == 0)
//			{
//				d = p1[0] - p0[0];
//			}
//			d = d + (p1[i] - p0[i]);
//		}
//
//		d = d / (int)p0.size();
//
//		T = estimateAffine2D(p0, p1);
//
//		// Extract traslation
//		//da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
//
//		transforms[0] = TransformParam(-d.x, -d.y, -atan2(T.at<double>(1, 0), T.at<double>(0, 0)));
//		iir(transforms, tau_stab);
//
//		if (roi.x + (int)transforms[1].dx < 0)
//		{
//			transforms[1].dx = double(1 - roi.x);
//		}
//		else if (roi.x + roi.width + (int)transforms[1].dx >= frame.cols)
//		{
//			transforms[1].dx = (double)(frame.cols - roi.x - roi.width);
//		}
//
//		if (roi.y + (int)transforms[1].dy < 0)
//		{
//			transforms[1].dy = (double)(1 - roi.y);
//		}
//		else if (roi.y + roi.height + (int)transforms[1].dy >= frame.rows)
//		{
//			transforms[1].dy = (double)(frame.rows - roi.y - roi.height);
//		}
//
//		//transforms[1].getTransform(T_stab, frame);
//		//transforms[1].getTransform(T_stab, a, b, c);
//		transforms[1].getTransform(T_stab, a, b, c, atan_ba);
//
//		if (wienner == false)
//		{
//			for (uint i = 0; i < p0.size(); i++)
//				circle(frame, p1[i], 2, colors[i], -1);
//		}
//
//
//
//		//frame_stabilized =  frame;
//
//
//		//~~~место для Винеровской фильтрации~~
//
//		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//		if (wienner == (true))
//		{
//			split(frame, channels);
//			LEN = sqrt(d.x * d.x + d.y * d.y) / Q;
//
//			if (d.x == 0.0)
//				if (d.y > 0.0)
//					THETA = 90.0;
//				else
//					THETA = -90.0;
//			else
//				THETA = atan(d.y / d.x) * 180 / 3.14159;
//			calcPSF(h, frame.size(), LEN, THETA); //смазывание в движении
//			calcWnrFilter(h, Hw, nsr);
//			//обработка трех цветных каналов
//			for (unsigned short i = 0; i < 3; i++)
//			{
//				channels[i].convertTo(channels[i], CV_32F);
//				filter2DFreq(channels[i], channels_wienner[i], Hw);
//
//			}
//
//			merge(channels_wienner, frame);
//			frame.convertTo(frame, CV_8U);
//		}
//
//		warpAffine(frame, frame_stabilized, T_stab, frame.size());
//		fixBorder(frame_stabilized, frame_part);
//		frame_stabilized.copyTo(cropped_img);
//		//printf("The time: %f seconds\n", seconds);
//
//		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//		//resize(cropped_img, cropped_img, Size(640, 480), 0.0, 0.0, INTER_AREA);
//		putText(cropped_img, format("Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), textOrg[0], fontFace, fontScale, color);
//		putText(cropped_img, format("harris_quality = %1.4f, Wienner %d", harris_quality, wienner), textOrg[1], fontFace, fontScale, color);
//		putText(cropped_img, format("[x y angle] [%2.1f : %2.1f : %1.2f]", T_stab.at<double>(0, 2), T_stab.at<double>(1, 2), 180.0/3.14*atan2(T_stab.at<double>(1, 0), T_stab.at<double>(0, 0))), textOrg[2], fontFace, fontScale, color);
//		putText(cropped_img, format("Tau stab[3][4] = %3.1f", tau_stab), textOrg[3], fontFace, fontScale, color);
//		putText(cropped_img, format("Frame cut-off part[ui] 1/%d, max_corners %d.", frame_part, max_corners), textOrg[4], fontFace, fontScale, color);
//		putText(cropped_img, format("fps = %2.1f", 1 / seconds), textOrg[6], fontFace, fontScale, color);
//		resize(cropped_img, cropped_img, Size(cropped_img.cols*2, cropped_img.rows * 2), 0.0, 0.0, INTER_CUBIC);
//
//		imshow("Esc for exit.", cropped_img);
//
//
//		int keyboard = waitKey(1);
//		if (keyboard == 'c')
//		{
//			imwrite("imgIn.jpg", frame);
//			imwrite("imgOut.jpg", cropped_img);
//		}
//		if (keyboard == 'q' || keyboard == 27)
//			break;
//		if (keyboard == '8')
//		{
//			nsr = nsr * 0.8;
//		}
//		if (keyboard == '7')
//		{
//			nsr = nsr * 1.25;
//		}
//		if (keyboard == '0')
//		{
//			wienner = false;
//		}
//		if (keyboard == '1')
//		{
//			wienner = true;
//		}
//		if (keyboard == '6')
//		{
//			if (Q < 20.0)
//				Q = Q * 1.05;
//		}
//		if (keyboard == '5')
//		{
//			if (Q > 1.0)
//				Q = Q * 0.95;
//			if (Q < 1.0)
//				Q = 1.0;
//		}
//		if (keyboard == '4')
//		{
//			if (tau_stab < 2000)
//				tau_stab = tau_stab * 2;
//		}
//		if (keyboard == '3')
//		{
//			if (tau_stab > 2)
//				tau_stab = tau_stab / 2;
//		}
//		if (keyboard == 'u' || keyboard == 'U')
//		{
//			if (frame_part < 10000)
//			{
//				frame_part = frame_part * 2;
//				roi.x = old_gray.cols / (frame_part * 2);
//				roi.y = old_gray.rows / (frame_part * 2);
//				roi.width = old_gray.cols * (frame_part - 1) / (frame_part);
//				roi.height = old_gray.rows * (frame_part - 1) / (frame_part);
//			}
//		}
//		if (keyboard == 'i' || keyboard == 'I')
//		{
//			if (frame_part > 2)
//			{
//				frame_part = frame_part / 2;
//				roi.x = old_gray.cols / (frame_part * 2);
//				roi.y = old_gray.rows / (frame_part * 2);
//				roi.width = old_gray.cols * (frame_part - 1) / (frame_part);
//				roi.height = old_gray.rows * (frame_part - 1) / (frame_part);
//			}
//
//		}
//
//		// Now update the previous frame and previous points
//		old_gray = frame_gray.clone();
//		p0 = good_new;
//	}
//	return 0;
//}
//
//
//
//void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
//{
//	Mat h(filterSize, CV_32F, Scalar(0));
//	Point point(filterSize.width / 2, filterSize.height / 2);
//	ellipse(h, point, Size(0, cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
//	Scalar summa = sum(h);
//	outputImg = h / summa[0];
//
//
//	Mat outputImg_norm;
//	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
//	imshow("PSF", outputImg_norm);
//}
//
//void calcPSF_circle(Mat& outputImg, Size filterSize, int len, double theta)
//{
//	Mat h(filterSize, CV_32F, Scalar(0));
//	Point point(filterSize.width / 2, filterSize.height / 2);
//	ellipse(h, point, Size(cvRound(float(len) / 2.0), cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
//	Scalar summa = sum(h);
//	outputImg = h / summa[0];
//
//
//	Mat outputImg_norm;
//	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
//	imshow("PSF", outputImg_norm);
//}
//
//
//void fftshift(const Mat& inputImg, Mat& outputImg)
//{
//	outputImg = inputImg.clone();
//	int cx = outputImg.cols / 2;
//	int cy = outputImg.rows / 2;
//	Mat q0(outputImg, Rect(0, 0, cx, cy));
//	Mat q1(outputImg, Rect(cx, 0, cx, cy));
//	Mat q2(outputImg, Rect(0, cy, cx, cy));
//	Mat q3(outputImg, Rect(cx, cy, cx, cy));
//	Mat tmp;
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//	q1.copyTo(tmp);
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//}
//
//void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
//{
//	Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
//	Mat complexI;
//	merge(planes, 2, complexI);
//	dft(complexI, complexI, DFT_SCALE);
//
//	Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
//	Mat complexH;
//	merge(planesH, 2, complexH);
//	Mat complexIH;
//	mulSpectrums(complexI, complexH, complexIH, 0);
//
//	idft(complexIH, complexIH);
//	split(complexIH, planes);
//	outputImg = planes[0];
//}
//
//void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
//{
//	Mat h_PSF_shifted;
//	fftshift(input_h_PSF, h_PSF_shifted);
//	Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
//	Mat complexI;
//	merge(planes, 2, complexI);
//	dft(complexI, complexI);
//	split(complexI, planes);
//	Mat denom;
//	pow(abs(planes[0]), 2, denom);
//	denom += nsr;
//	divide(planes[0], denom, output_G);
//}
//
//void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
//{
//	int Nx = inputImg.cols;
//	int Ny = inputImg.rows;
//	Mat w1(1, Nx, CV_32F, Scalar(0));
//	Mat w2(Ny, 1, CV_32F, Scalar(0));
//
//	float* p1 = w1.ptr<float>(0);
//	float* p2 = w2.ptr<float>(0);
//	float dx = float(2.0 * CV_PI / Nx);
//	float x = float(-CV_PI);
//	for (int i = 0; i < Nx; i++)
//	{
//		p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
//		x += dx;
//	}
//	float dy = float(2.0 * CV_PI / Ny);
//	float y = float(-CV_PI);
//	for (int i = 0; i < Ny; i++)
//	{
//		p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
//		y += dy;
//	}
//	Mat w = w2 * w1;
//	multiply(inputImg, w, outputImg);
//}