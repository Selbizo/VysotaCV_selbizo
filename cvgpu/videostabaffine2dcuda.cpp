//#include <iostream>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/video.hpp>
//#include "testGpuFunctions.hpp"
//using namespace cv;
//using namespace std;
//
//int main(int argc, char* argv[])
//{
//    int R = 3;
//    int KernelSize = 3;
//    double nsr = 0.004;
//    int dx = 1;
//    int dy = 1;
//    double Q = 1.0;
//    double LEN = 0.0;
//    double THETA = 0.0;
//
//
//    //Для отображения надписей на кадре
//    setlocale(LC_ALL, "RU");
//    Point textOrg(10, 80);
//    int fontFace = FONT_HERSHEY_PLAIN;
//    double fontScale = 1;
//    Scalar color(255, 255, 255);
//
//    string strInFileName = "imgInAustralia.jpg";
//    
//    cv::Ptr <cuda::Filter> gaussFilterFloat = cuda::createGaussianFilter(CV_32F, CV_32F, Size(5, 5), 1.5, 1.5);
//
//    Mat imgIn, imgOut, imgInGray, imgOutGray;
//    imgIn = imread(strInFileName);
//    
//    cuda::GpuMat gImgIn, gImgOut, gHw, gH, gFrame;
//    vector<cuda::GpuMat> gChannels(3);
//	vector<cuda::GpuMat> gChannelsWienner(3);
//	cuda::GpuMat gFrameWienner;
//    gImgIn.upload(imgIn);
//
//    Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
//    Mat Hw, h;
//    while (true)
//        {
//        LEN = sqrt(dx * dx + dy * dy) / Q;
//        			if (dx == 0.0)
//        				if (dy > 0.0)
//        					THETA = 90.0;
//        				else
//        					THETA = -90.0;
//        			else
//        				THETA = atan(dy / dx) * 180 / 3.14159;
//
//                    gImgIn.copyTo(gFrame);
//                    //gaussFilter->apply(gImgIn, gFrame);
//        			cuda::split(gFrame, gChannels);
//        			GcalcPSF(gH, gFrame.size(), Size((int)LEN + 100, (int)LEN + 100), LEN, THETA);
//        			//GcalcPSFCircle(gH, gFrame.size(), LEN, THETA);
//        			GcalcWnrFilter(gH, gHw, nsr);
//        
//        			for (unsigned short i = 0; i < 3; i++) //обработка трех цветных каналов
//        			{
//        				gChannels[i].convertTo(gChannels[i], CV_32F);
//                        gaussFilterFloat->apply(gChannels[i], gChannels[i]);
//        				Gfilter2DFreq(gChannels[i], gChannelsWienner[i], gHw);
//        			}
//        			cuda::merge(gChannelsWienner, gFrame);
//        			gFrame.convertTo(gFrame, CV_8U);
//
//
//            //cuda::resize(gFrame, gFrame, Size(gFrame.cols, gFrame.rows), 0.0, 0.0, INTER_AREA);
//            cuda::bilateralFilter(gFrame, gFrame, 31, 5.0, 10.0);
//            gFrame.download(imgOut);
//
//            //CPU
//        putText(imgOut, format("dx[a][d] = %d, dy [s][w]= %d, SNR[7][8] = %3.3f, Q[q][e] = %3.3f", dx,  dy, 1 / nsr, Q), textOrg, fontFace, fontScale, color);
//        imshow("imgOut", imgOut);
//
//        //Вычисляем гистограмму
//		int histSize = 256; // Количество уровней яркости (0-255)
//		float range[] = { 0, 256 }; // Диапазон значений пикселей
//		const float* ranges[] = { range };
//
//		Mat hist, hist_filtered;
//
//        cv::cvtColor(imgIn, imgInGray, COLOR_BGR2GRAY);
//        cv::cvtColor(imgOut, imgOutGray, COLOR_BGR2GRAY);
//		calcHist(&imgInGray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
//		calcHist(&imgOutGray, 1, 0, Mat(), hist_filtered, 1, &histSize, ranges, true, false);
//		// Нормализуем гистограмму
//		normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
//		normalize(hist_filtered, hist_filtered, 0, 255, NORM_MINMAX, -1, Mat());
//		// Рисуем гистограмму
//		int hist_w = 512; // Ширина окна гистограммы
//		int hist_h = 400; // Высота окна гистограммы
//		int bin_w = cvRound((double)hist_w / histSize); // Ширина одного столбца
//
//		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
//		Mat histFilteredImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
//
//		for (int i = 0; i < histSize; i++) {
//			rectangle(
//				histImage,
//				Point(bin_w * i, hist_h),
//				Point(bin_w * (i + 1), hist_h - cvRound(hist.at<float>(i))),
//				Scalar(255, 200, 255),
//				FILLED
//			);
//		}
//
//		for (int i = 0; i < histSize; i++) {
//			rectangle(
//				histFilteredImage,
//				Point(bin_w * i, hist_h),
//				Point(bin_w * (i + 1), hist_h - cvRound(hist_filtered.at<float>(i))),
//				Scalar(255, 255, 200),
//				FILLED
//			);
//		}
//
//		// Показываем гистограмму
//		namedWindow("Filtered Grayscale Histogram", WINDOW_AUTOSIZE);
//		imshow("Filtered Grayscale Histogram", histFilteredImage);
//
//		namedWindow("Grayscale Histogram", WINDOW_AUTOSIZE);
//		imshow("Grayscale Histogram", histImage);
//
//
//        int keyboard = waitKey(1);
//        if (keyboard == '8')
//        {
//            nsr = nsr * 0.8;
//        }
//        if (keyboard == '7')
//        {
//            nsr = nsr * 1.25;
//        }
//
//        if (keyboard == 'w')
//        {
//            dy++;
//        }
//        if (keyboard == 's')
//        {
//            dy--;
//        }
//        if (keyboard == 'e')
//        {
//            Q+=0.1;
//        }
//        if (keyboard == 'q' && Q > 1.0)
//        {
//            Q-=0.1;
//        }
//        if (keyboard == 'd')
//        {
//            dx++;
//        }
//        if (keyboard == 'a')
//        {
//            //if (dx > 0)
//                dx--;
//        }
//        if (keyboard == 'r')
//        {
//            dx = 50;
//            dy = 50;
//        }
//
//        if (keyboard == 27)
//        {
//            imwrite("imgIn.jpg", imgIn);
//            imwrite("imgOut.jpg", imgOut);
//            break;
//        }
//        if (keyboard == 'c')
//        {
//
//            //imwrite(format("INPUT KernelSize = %d, SNR = %3.3f, R = %d.jpg", KernelSize, 1 / nsr, R), imgIn);
//            imwrite(format("INPUT_BLURED KernelSize = %d, SNR = %3.3f, R = %d.jpg", KernelSize, 1 / nsr, R), imgIn);
//            imwrite(format("OUTPUT KernelSize = %d, SNR = %3.3f, R = %d.jpg", KernelSize, 1 / nsr, R), imgOut);
//        }
//
//    }
//
//    return 0;
//}
//
///*
//void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
//{
//    Mat h(filterSize, CV_32F, Scalar(0));
//    Point point(filterSize.width / 2, filterSize.height / 2);
//    ellipse(h, point, Size(0, cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(42), FILLED);
//    ellipse(h, point, Size(0, cvRound(float(len) / 4.0)), 90.0 - theta, 0, 360, Scalar(162), FILLED);
//    ellipse(h, point, Size(0, cvRound(float(len) / 6.0)), 90.0 - theta, 0, 360, Scalar(210), FILLED);
//    ellipse(h, point, Size(0, cvRound(float(len) / 8.0)), 90.0 - theta, 0, 360, Scalar(240), FILLED);
//    ellipse(h, point, Size(0, cvRound(float(len) / 12.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
//    Scalar summa = sum(h);
//    Mat outputImg_norm;
//    h.copyTo(outputImg_norm);
//    outputImg = h / summa[0];
//
//
//    normalize(h, outputImg_norm, 0, 255, NORM_MINMAX);
//
//    outputImg_norm(Rect(filterSize.width*4/9, filterSize.height*4 / 9, filterSize.width / 9, filterSize.height / 9)).copyTo(outputImg_norm);
//    resize(outputImg_norm, outputImg_norm, Size(filterSize.width, filterSize.height), 0.0, 0.0, INTER_LINEAR);
//    imshow("PSF", outputImg_norm);
//
//}
//
//void calcPSFafter(Mat& outputImg, Size filterSize, int len, double theta)
//{
//    Mat h(filterSize, CV_32F, Scalar(0));
//    Point point(filterSize.width / 2, filterSize.height / 2);
//    ellipse(h, point, Size(0, cvRound(float(1.0) / 2.0)), 90.0 - theta, 0, 360, Scalar(100), FILLED);
//    Point point2(filterSize.width / 2+len*cos(theta*3.14/180.0), filterSize.height / 2 + len * sin(theta*3.14/180.0));
//    ellipse(h, point2, Size(0, cvRound(float(1.0) / 2.0)), 90.0 - theta, 0, 360, Scalar(100), FILLED);
//    //ellipse(h, point, Size(0, cvRound(float(len) / 4.0)), 90.0 - theta, 0, 360, Scalar(150), FILLED);
//    //ellipse(h, point, Size(0, cvRound(float(len) / 8.0)), 90.0 - theta, 0, 360, Scalar(200), FILLED);
//    //ellipse(h, point, Size(0, cvRound(float(len) / 12.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
//    Scalar summa = sum(h);
//    Mat outputImg_norm;
//    h.copyTo(outputImg_norm);
//    outputImg = h / summa[0];
//
//
//    normalize(h, outputImg_norm, 0, 255, NORM_MINMAX);
//
//    outputImg_norm(Rect(filterSize.width * 4 / 9, filterSize.height * 4 / 9, filterSize.width / 9, filterSize.height / 9)).copyTo(outputImg_norm);
//    resize(outputImg_norm, outputImg_norm, Size(filterSize.width, filterSize.height), 0.0, 0.0, INTER_LINEAR);
//    imshow("PSF", outputImg_norm);
//
//}
//
//void fftshift(const Mat& inputImg, Mat& outputImg)
//{
//    outputImg = inputImg.clone();
//    int cx = outputImg.cols / 2;
//    int cy = outputImg.rows / 2;
//    Mat q0(outputImg, Rect(0, 0, cx, cy));
//    Mat q1(outputImg, Rect(cx, 0, cx, cy));
//    Mat q2(outputImg, Rect(0, cy, cx, cy));
//    Mat q3(outputImg, Rect(cx, cy, cx, cy));
//    Mat tmp;
//    q0.copyTo(tmp);
//    q3.copyTo(q0);
//    tmp.copyTo(q3);
//    q1.copyTo(tmp);
//    q2.copyTo(q1);
//    tmp.copyTo(q2);
//}
//
//void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
//{
//    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
//    Mat complexI;
//    merge(planes, 2, complexI);
//    dft(complexI, complexI, DFT_SCALE);
//
//    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
//    Mat complexH;
//    merge(planesH, 2, complexH);
//    Mat complexIH;
//    mulSpectrums(complexI, complexH, complexIH, 0);
//
//    idft(complexIH, complexIH);
//    split(complexIH, planes);
//    outputImg = planes[0];
//}
//
//void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
//{
//    Mat h_PSF_shifted;
//    fftshift(input_h_PSF, h_PSF_shifted);
//    Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
//    Mat complexI;
//    merge(planes, 2, complexI);
//    dft(complexI, complexI);
//    split(complexI, planes);
//    Mat denom;
//    pow(abs(planes[0]), 2, denom);
//    denom += nsr;
//    divide(planes[0], denom, output_G);
//    imshow("Wnr_filter", output_G);
//}
//
//void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
//{
//    int Nx = inputImg.cols;
//    int Ny = inputImg.rows;
//    Mat w1(1, Nx, CV_32F, Scalar(0));
//    Mat w2(Ny, 1, CV_32F, Scalar(0));
//
//    float* p1 = w1.ptr<float>(0);
//    float* p2 = w2.ptr<float>(0);
//    float dx = float(2.0 * CV_PI / Nx);
//    float x = float(-CV_PI);
//    for (int i = 0; i < Nx; i++)
//    {
//        p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
//        x += dx;
//    }
//    float dy = float(2.0 * CV_PI / Ny);
//    float y = float(-CV_PI);
//    for (int i = 0; i < Ny; i++)
//    {
//        p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
//        y += dy;
//    }
//    Mat w = w2 * w1;
//    multiply(inputImg, w, outputImg);
//}*/
