//Алгоритм стабилизации видео на основе вычисление Lucas-Kanade Optical Flow

#include "testGpuFunctions.hpp"

using namespace cv;
using namespace std;

//int main(int argc, char** argv)



int main()
{
	//string videoSource = "Timeline 3 480p.mp4";
	//string videoSource = "live_7_small_3.avi";
	//string videoSource = "http://192.168.0.102:4747/video"; // pad6-100, pixel4-101, pixel-102
	//string videoSource = "http://10.108.144.71:4747/video"; // pad6-100, pixel4-101, pixel-102
	int videoSource = 0;

	bool writeVideo = true;
	bool stabPossible = false;

	// Create some random colors
	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < 10000; i++)
	{
		int b = rng.uniform(0, 250);
		int g = rng.uniform(0, 200);
		int r = rng.uniform(0, 200);
		colors.push_back(Scalar(b, g, r));
	}
	// переменные для поиска характерных точек
	int n = 1; //коэффициент сжатия изображения для обработки
	vector<uchar> status;
	//vector<Point2f> err;
	Mat err;


	int	srcType = CV_8UC1;
	int maxCorners = 400 / n;      //100/n
	double qualityLevel = 0.02; //0.0001
	double minDistance = 8.0; //8.0
	int blockSize = 45; //45 80 максимальное значение окна
	bool useHarrisDetector = true;
	double harrisK = qualityLevel;

	Ptr<cuda::CornersDetector > d_features = cv::cuda::createGoodFeaturesToTrackDetector(srcType,
		maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, harrisK);


	// переменные для оптического потока
	bool useGray = true;
	int winSize = blockSize;
	int maxLevel = 5;
	int iters = 10;
	double minDist = minDistance;
	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(
		cv::Size(winSize, winSize), maxLevel, iters);

	//переменные для запоминания кадров и характерных точек
	Mat frame, frameGray, temp;
	Mat oldFrame, oldGray;

	vector<Point2f> p0, p1, good_new;
	cuda::GpuMat gP0, gP1;
	cuda::GpuMat gFrame, gFrameGray;
	cuda::GpuMat gOldFrame, gOldGray;
	cuda::GpuMat gStatus, gErr;

	Point2f d = Point2f(0.0f, 0.0f);
	Mat T;
	Mat TStab(2, 3, CV_64F);

	Mat T3d;
	Mat T3dStab(3, 4, CV_64F);

	Mat frameStabilized, frame_out;

	cuda::GpuMat gT;
	cuda::GpuMat gTStab(2, 3, CV_64F);
	cuda::GpuMat gFrameStabilized, gFrameOut;

	double tauStab = 100.0;
	double kSwitch = 0.001;
	double framePart = 0.8;
	Rect roi;

	vector <TransformParam> transforms(4);
	for (int i = 0; i < 4;i++)
	{
		transforms[i].dx = 0.0;
		transforms[i].dy = 0.0;
		transforms[i].da = 0.0;
	}
	

	// переменные для фильтра Виннера
	//cv::Ptr <cuda::Filter> gaussFilter = cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(31, 31), 1.0, 1.0);
	cv::Ptr <cuda::Filter> laplasianFilterGray = cuda::createLaplacianFilter(CV_8UC1, CV_8UC1);
	cv::Ptr <cuda::Filter> maxBoxFilterGray = cuda::createBoxMaxFilter(CV_8UC1, cv::Size(3, 3));
	cv::Ptr <cuda::Filter> highPassFilterBGR = cuda::createSobelFilter(CV_8UC3, CV_8UC3, 1, 1, 1, 7);
	cv::Ptr <cuda::Filter> highPassFilterGray = cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 1, 1);


	cv::Ptr <cuda::CannyEdgeDetector> cannyEdgeDet = cuda::createCannyEdgeDetector(10.0, 20.0, 3, false);


	Mat Hw, h, frameGray_wienner;
	cuda::GpuMat gHw, gH, gFrameGrayWienner;

	bool wienner = false;
	bool threadwiener = false;
	double nsr = 0.01;
	double Q = 8.0; // скважность считывания кадра на камере (выдержка к частоте кадров) (умножена на 10)
	double LEN = 0;
	double THETA = 0.0;

	//для обработки трех каналов по Виннеру
	vector<Mat> channels(3);
	vector<Mat> channelsWienner(3);
	Mat frame_wienner;



	vector<cuda::GpuMat> gChannels(3);
	vector<cuda::GpuMat> gChannelsWienner(3);
	cuda::GpuMat gFrameWienner;



	//для вывода изображения на дисплей
	Mat croppedImg, frame_crop;
	cuda::GpuMat gCroppedImg, gFrameCrop, gFrameCropHigh;



	// ~~ для счетчика кадров в секунду ~~~~~~~~~~~~~~~~~~~~~~//
	int frameCnt = 0;
	double seconds = 0.0;
	double secondsPing = 0.0;
	clock_t start = clock();
	clock_t end = clock();

	clock_t startPing = clock();
	clock_t endPing = clock();

	//~Захват первого кадра ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

	VideoCapture capture(videoSource);

	// Попытка установить 720p (1280x720)

	//capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	//capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);


	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to connect camera!" << endl;
		return 0;
	}

	capture >> oldFrame;

	const double a = oldFrame.cols;
	const double b = oldFrame.rows;
	const double c = sqrt(a * a + b * b);
	const double atan_ba = atan2(b, a);

	roi.x = a * ((1.0 - framePart) / 2.0);
	roi.y = b * ((1.0 - framePart) / 2.0);
	roi.width = a * framePart;
	roi.height = b * framePart;

	// Параметры фильтра Калмана
	int stateSize = 2;           // Размер состояния (x, v)
	int measSize = 1;            // Размер измерения (x)
	int contrSize = 0;           // Размер управления (нет управления)

	Mat greenLine(cv::Size(a, b), CV_8UC3, Scalar(0, 0, 0));
	Point point(a / 2, b / 2);

	//Для отображения надписей на кадре
	setlocale(LC_ALL, "RU");
	vector <Point> textOrg(10);
	vector <Point> textOrgCrop(10);
	vector <Point> textOrgStab(10);
	vector <Point> textOrgOrig(10);
	if (writeVideo)
	{ 
		for (int i = 0; i < 10; i++)
		{
			textOrg[i].x = 10 + a;
			textOrg[i].y = 10 + 35 * (i + 1) + b;

			textOrgCrop[i].x = 10;
			textOrgCrop[i].y = 10 + 35 * (i + 1) + b;

			textOrgStab[i].x = 10;
			textOrgStab[i].y = 10 + 35 * (i + 1);

			textOrgOrig[i].x = 10 + a;
			textOrgOrig[i].y = 10 + 35 * (i + 1);
		}
	}
	else {
		for (int i = 0; i < 10; i++)
		{
			textOrg[i].x = 20;
			textOrg[i].y = 100 + 40 * (i + 1);
		}
	}


	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1.4;
	Scalar color(40, 140, 0);


	line(greenLine, Point(a / 2, b * 5 / 9), Point(a / 2, b), Scalar(0, 255, 0), 2);

	line(greenLine, Point(5 * a / 9, b / 2), Point(a, b / 2), Scalar(0, 255, 0), 2);
	line(greenLine, Point(0, b / 2), Point(4 * a / 9, b / 2), Scalar(0, 255, 0), 2);
	cuda::GpuMat gGreenLine;
	gGreenLine.upload(greenLine);

	// Создадим маску по умолчанию ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Mat mask_host = Mat::zeros(cv::Size(a / n, b / n), CV_8U);
	rectangle(mask_host, Rect(a * (1.0 - 0.8) / 2 / n, b * (1.0 - 0.8) / 2 / n, a * 0.8 / n, b * 0.8 / n), Scalar(255), FILLED); // Прямоугольная маска
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	cuda::GpuMat mask_device(mask_host);

	// Создаем GpuMat для мнимой части фильтра Винера
	cuda::GpuMat zeroMatH(cv::Size(a, b), CV_32F, Scalar(0));
	cuda::GpuMat complexH;
	Ptr<cuda::DFT> forwardDFT = cuda::createDFT(cv::Size(a, b), DFT_SCALE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> inverseDFT = cuda::createDFT(cv::Size(a, b), DFT_INVERSE | DFT_COMPLEX_INPUT);


	// Создаем объект для записи отклонения в CSV файл
	std::ofstream outputFile("output.txt");
	if (!outputFile.is_open())
	{
		cout << "Не удалось открыть файл для записи" << endl;
		return -1;
	}

	// Запись заголовка в CSV файл
	outputFile << "FrameNumber,dx,dy,X,Y" << endl;
	int frameCount = 0;

	while (true) {
		initFirstFrame(capture, oldFrame, gOldFrame, gOldGray, gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms, n, kSwitch, a, b, mask_device, stabPossible);
		if (stabPossible)
			break;
	}


	//~~~~~~~~создание объекта записи видео
	VideoWriter writer;
	VideoWriter writerSmall;
	cv::Mat writerFrame(oldFrame.rows * 2, oldFrame.cols * 2, CV_8UC3);
	cv::Mat writerFrameSmall(oldFrame.rows, oldFrame.cols, CV_8UC3);
	if (writeVideo) {
		bool isColor = (oldFrame.type() == CV_8UC3);
		//--- INITIALIZE VIDEOWRITER

		int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)

		double fps = 30.0;                          // framerate of the created video stream
		string filename = "./live_7.avi";             // name of the output video file
		string filenameSmall = "./live_7_small.avi";  // name of the output video file

		writer.open(filename, codec, fps, writerFrame.size(), isColor);
		if (!writer.isOpened()) {
			cerr << "Could not open the output video file for write\n";
			return -1;
		}

		writerSmall.open(filenameSmall, codec, fps, writerFrameSmall.size(), isColor);
		if (!writerSmall.isOpened()) {
			cerr << "Could not open the output video file for writeSmall\n";
			return -1;
		}
	}

	while (true) {
		//нужно только для iirAdaptive

			   //readFrameFromCapture(capture, frame);
		++frameCount;
		secondsPing = 0.95 * secondsPing + 0.05 * (double)(endPing - startPing) / CLOCKS_PER_SEC;

		//std::thread readFromCapThread(readFrameFromCapture, &capture, &frame);
		if (stabPossible) {
			p0.clear();
			for (uint i = 0; i < p1.size(); ++i)
			{
				if (status[i] && p1[i].x < a * 31 / 32 && p1[i].x > a * 1 / 32 && p1[i].y < b * 31 / 32 && p1[i].y > b * 1 / 16) {
					//if (status[i] && p1[i].x < a * 3 / 4 && p1[i].x > a * 1 / 7 && p1[i].y < b * 6 / 7 && p1[i].y > b * 1 / 4) {
					p0.push_back(p1[i]); // Выбор точек good_new
				}
			}

			//if (p1.size() < maxCorners / 4 && rng.uniform(0.0, 1.0) > 0.98)
			//{
			//	p0.push_back(Point2f(a / 2 + rng.uniform(-a / 3, a / 3), b / 2 + rng.uniform(-b/3, b/3)));
			//}

			gFrameGray.copyTo(gOldGray);
			gP0.upload(p0);
			if (kSwitch < 0.01)
				kSwitch = 0.01;
			if (kSwitch < 1.0)
			{
				kSwitch *= 1.04;
				kSwitch += 0.005;

			}

			else if (kSwitch > 1.0)
				kSwitch = 1.0;

			capture >> frame; //нужно для .upload(frame)


		}

		startPing = clock();
		if (frameCnt == 200)
		{

			end = clock();
			seconds = (double)(end - start) / CLOCKS_PER_SEC / (frameCnt + 1);
			frameCnt = 0;
			start = clock();
		}

		//условно thread capRead.join();  
		//readFromCapThread.join();

		if (frame.empty())
		{
			capture.release();
			capture = VideoCapture(videoSource);
			capture >> frame;

		}


		if (writeVideo && stabPossible) {
			//cv::putText(frame, format("Raw Video Stream"), textOrg[9], fontFace, fontScale, color, 2, 8, false);
			rectangle(writerFrame, Rect(a, b, a, b), Scalar(0, 0, 0), FILLED); // Прямоугольная маска
			frame.copyTo(writerFrame(cv::Rect(a, 0, a, b))); //original video
		}
		frameCnt++;

		if (stabPossible) {

			if (wienner == false && p0.size() > 0 && 0)
			{
				for (uint i = 0; i < p0.size(); i++)
					circle(frame, p1[i], 4, colors[i], -1);
			}

			gFrame.upload(frame);

			cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, 3);
			//cuda::bilateralFilter(gFrame, gFrame, 7, 3.0, 1.0);

			cuda::cvtColor(gFrame, gFrameGray, COLOR_BGR2GRAY);
			cuda::bilateralFilter(gFrameGray, gFrameGray, 7, 3.0, 1.0);
		}

		if ((gP0.cols < maxCorners * 1 / 5) || !stabPossible)
		{
			//cerr << "Unable to find enough corners, nessessary " << maxCorners / 4 << "." << endl;

			if (maxCorners > 300)
				maxCorners *= 0.95;
			if (gP0.cols < maxCorners * 1 / 4 && stabPossible)
				d_features->setMaxCorners(maxCorners);
			p0.clear();
			p1.clear();
			gOldGray.release();
			//if (stabPossible)
			//{
			capture >> frame;
			if (!stabPossible) {
				rectangle(writerFrame, Rect(a, b, a, b), Scalar(0, 0, 0), FILLED); // Прямоугольная маска
				frame.copyTo(writerFrame(cv::Rect(a, 0, a, b))); //original video
			}
			gFrame.upload(frame);
			//}

			//cuda::bilateralFilter(gFrame, gFrame, 7, 3.0, 1.0);

			cuda::cvtColor(gFrame, gFrameGray, COLOR_BGR2GRAY);

			cuda::resize(gFrameGray, gFrameGray, cv::Size(a / n, b / n), 0.0, 0.0, 3);


			if (frameCnt % 10 == 1 && !stabPossible)
			{
				initFirstFrame(capture, oldFrame, gOldFrame, gOldGray, gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms, n, kSwitch, a, b, mask_device, stabPossible); //70ms
				//stabPossible = 0;
			}
			else
				initFirstFrameZero(oldFrame, gOldFrame, gOldGray, gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms, n, kSwitch, a, b, mask_device, stabPossible); //70ms


			if (stabPossible) {
				d_pyrLK_sparse->calc(gOldGray, gFrameGray, gP0, gP1, gStatus, gErr);
				gP1.download(p1);
				gErr.download(err);
			}
			//cout << "err.Size() = " << err.Size();

		}
		else if (stabPossible) {
			cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, 3);
			cuda::resize(gFrameGray, gFrameGray, cv::Size(a / n, b / n), 0.0, 0.0, 3);

			d_pyrLK_sparse->calc(useGray ? gOldGray : gOldFrame, useGray ? gFrameGray : gFrame, gP0, gP1, gStatus);
			//cuda::multiply(gP1, 4.0, gP1);
			gP1.download(p1);
		}

		if ((gP1.cols > maxCorners * 4 / 5) && stabPossible) { //обновление критериев поиска точек
			{
				maxCorners *= 1.02;
				maxCorners += 1;
				d_features->setMaxCorners(maxCorners);
			}
		}
		if (stabPossible) {
			download(gStatus, status);

			// условно gFrameOpticalFlow.join();
			getBiasAndRotation(p0, p1, d, transforms, T, n, T3d); //уже можно делать Винеровскую фильтрацию

			if (T.rows == 2 && T.cols == 3)
			{
				double xDev = T.at<double>(0, 2);
				double yDev = T.at<double>(1, 2);

				// Запись отклонения в CSV файл
				//outputFile << frameCount << "\t" << xDev << "\t" << yDev << "\t" << transforms[1].dx <<"\t" << transforms[1].dy << endl;
				outputFile << frameCount << "," << round(xDev * 100.0) / 100.0 << "," << round(yDev * 100.0) / 100.0 << "," << round(transforms[1].dx * 100.0) / 100.0 << "," << round(transforms[1].dy * 100.0) / 100.0 << endl;
			}

			iirAdaptive(transforms, tauStab, roi, gFrame.cols, gFrame.rows, kSwitch);
			transforms[1].getTransform(TStab, a, b, c, atan_ba, framePart);
			//transforms[2].getTransform(TStab, a, b, c, atan_ba, framePart);

			// Винеровская фильтрация
			if (wienner && kSwitch > 0.01)
			{
				//LEN = sqrt(d.x * d.x + d.y * d.y) / Q;
				LEN = sqrt(transforms[0].dx * transforms[0].dx + transforms[0].dy * transforms[0].dy) / Q;
				if (transforms[0].dx == 0.0)
					if (transforms[0].dy > 0.0)
						THETA = 90.0;
					else
						THETA = -90.0;
				else
					THETA = atan(transforms[0].dy / transforms[0].dx) * 180.0 / 3.14159;

				cuda::bilateralFilter(gFrame, gFrame, 13, 5.0, 3.0);
				gFrame.convertTo(gFrame, CV_32F);
				cuda::split(gFrame, gChannels);

				GcalcPSF(gH, gFrame.size(), cv::Size((int)LEN * 1 + 10, (int)LEN * 1 + 10), LEN, THETA);
				GcalcWnrFilter(gH, gHw, nsr);

				// Объединяем действительную и мнимую часть фильтра в комплексную матрицу
				vector<cuda::GpuMat> planesH = { gHw, zeroMatH };

				cuda::merge(planesH, complexH);
				if (!threadwiener)
				{
					for (unsigned short i = 0; i < 3; i++) //обработка трех цветных каналов можно разделить на три потока
					{
						//gChannels[i].convertTo(gChannels[i], CV_32F);
						Gfilter2DFreqV2(gChannels[i], gChannelsWienner[i], complexH, forwardDFT, inverseDFT);
					}
				}
				else
				{
					std::thread blueChannelWiener(channelWiener, &gChannels[0], &gChannelsWienner[0], &complexH, &forwardDFT, &inverseDFT);
					std::thread greenChannelWiener(channelWiener, &gChannels[1], &gChannelsWienner[1], &complexH, &forwardDFT, &inverseDFT);
					std::thread redChannelWiener(channelWiener, &gChannels[2], &gChannelsWienner[2], &complexH, &forwardDFT, &inverseDFT);

					blueChannelWiener.join();
					greenChannelWiener.join();
					redChannelWiener.join();
				}
				cuda::merge(gChannelsWienner, gFrame);
				gFrame.convertTo(gFrame, CV_8UC3);
				cuda::bilateralFilter(gFrame, gFrame, 13, 5.0, 3.0);
			}



			//cuda::addWeighted(gFrame, 1.0, gGreenLine, 0.3, 1.0, gFrame);
			//cuda::add(gFrame, gGreenLine, gFrame);

			cuda::warpAffine(gFrame, gFrameStabilized, TStab, cv::Size(a, b));


			gFrameCrop = gFrameStabilized(roi); //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			//gFrameCrop = gFrameStabilized;

			//cuda::cvtColor(gFrameCrop, gFrameCropHigh, COLOR_BGR2GRAY);
			//gFrameCropHigh.convertTo(gFrameCropHigh, CV_8UC1);
			////highPassFilterBGR->apply(gFrameCrop, gFrameCropHigh);
			//laplasianFilterGray->apply(gFrameCropHigh, gFrameCropHigh);
			//maxBoxFilterGray->apply(gFrameCropHigh, gFrameCropHigh);
			//cuda::cvtColor(gFrameCropHigh, gFrameCropHigh, COLOR_GRAY2BGR);
			//cuda::addWeighted(gFrameCrop, 1.0, gFrameCropHigh, 0.3, 0.8, gFrameCrop);



			cuda::resize(gFrameCrop, gFrameCrop, cv::Size(a, b), 0.0, 0.0, 3);
			gFrameCrop.download(croppedImg);
			endPing = clock();

			
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			// Вывод изображения на дисплей
			if (writeVideo)
			{
				gFrame = gFrame(roi);
				cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, 3);
				gFrame.download(frame);


				frame.copyTo(writerFrame(cv::Rect(0, frame.rows, frame.cols, frame.rows)));

				croppedImg.copyTo(writerFrame(cv::Rect(0, 0, frame.cols, frame.rows)));


				cv::putText(writerFrame, format("-_-Wiener Filter Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), textOrg[0], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("WienerIsOn[1] %d, threadsIsOn[t] %d, stabIsOn %d", wienner, threadwiener, stabPossible), textOrg[1], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("[X Y Roll] %2.2f %2.2f %2.2f]", transforms[1].dx, transforms[1].dy, transforms[1].da*RAD_TO_DEG), textOrg[3], fontFace, fontScale, color, 2, 8, false);
				//cv::putText(writerFrame, format("[dx dy] %2.2f %2.2f]", d.x, d.y), textOrg[3], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("[dX dY dRoll] %2.2f %2.2f %2.2f]", transforms[0].dx, transforms[0].dy, transforms[0].da), textOrg[2], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("[skoX skoY skoRoll] %2.2f %2.2f %2.2f]", transforms[2].dx, transforms[2].dy, transforms[2].da), textOrg[8], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Filter time[3][4]= %3.0f frames, filter power = %1.2f", tauStab, kSwitch), textOrg[4], fontFace, fontScale/1.2, color, 2, 8, false);
				cv::putText(writerFrame, format("Crop[w][s] = %2.2f, %d Current corners of %d.", 1 / framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, Scalar(120, 60, 255), 2, 8, false);
				cv::putText(writerFrame, format("FPS = %2.1f, Ping = %1.3f.", 1 / seconds, secondsPing), textOrg[6], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Image resolution: %3.0f x %3.0f.", a, b), textOrg[7], fontFace, fontScale / 1.2, color, 2, 8, false);

				cv::putText(writerFrame, format("Crop Stab OFF"), textOrgCrop[9], fontFace, fontScale * 1.5, Scalar(30, 30, 200), 2, 8, false);
				cv::putText(writerFrame, format("Original video"), textOrgOrig[0], fontFace, fontScale * 1.5, Scalar(255, 20, 200), 2, 8, false);
				cv::putText(writerFrame, format("Crop Stab ON"), textOrgStab[9], fontFace, fontScale * 1.5, Scalar(20, 200, 20), 2, 8, false);
				cv::putText(writerFrame, format("atan: %3.3f x %3.3f.", framePart * (a - c * cos(atan(b / a) - transforms[1].da)) / 2, framePart * (b - c * sin(atan(b / a) - transforms[1].da)) / 2), textOrgStab[7], fontFace, fontScale / 1.2, color, 2, 8, false);
				
				if (p0.size() > 0)
					for (uint i = 0; i < p0.size(); i++)
						circle(writerFrame, cv::Point2f(p1[i].x + a, p1[i].y), 4, colors[i], -1); //circle(writerFrame, p1[i], 2, colors[i], -1);


				
				cv::ellipse(writerFrame, cv::Point2f(-transforms[1].dx + a + a / 2, -transforms[1].dy + b / 2), cv::Size(a * framePart / 2, 0), 0.0 - 0.0*transforms[1].da * RAD_TO_DEG, 0, 360, Scalar(200, 20, 80), 2);
				cv::ellipse(writerFrame, cv::Point2f(-transforms[1].dx + a + a / 2, -transforms[1].dy + b / 2), cv::Size(0, b * framePart / 2), 0.0 - 0.0*transforms[1].da * RAD_TO_DEG, 0, 360, Scalar(200, 20, 80), 2);
				
				//ellipse(h, point, cv::Size(0, cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
				cv::imshow("Writed", writerFrame);
				writer.write(writerFrame);




				writerSmall.write(croppedImg);

			}
			else {
				cv::putText(croppedImg, format("2) Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), textOrg[0], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("Wienner[1] %d, thread[t] %d", wienner, threadwiener), textOrg[1], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("[x y angle] %2.0f %2.0f %1.1f]", TStab.at<double>(0, 2), TStab.at<double>(1, 2), 180.0 / 3.14 * atan2(TStab.at<double>(1, 0), TStab.at<double>(0, 0))),
					textOrg[2], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("[dx dy] %2.2f %2.2f]", d.x, d.y), textOrg[3], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("Tau[3][4] = %3.0f, kSwitch = %1.2f", tauStab, kSwitch), textOrg[4], fontFace, fontScale, color, 2, 8, false);
				//cv::putText(croppedImg, format("Cut-off[w][s] 1/%2.2f, %d Corners of %d.", framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Crop[w][s] = %2.1f, %d Corners of %d.", 1 / framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("fps = %2.1f, ping = %1.3f, Size: %d x %d.", 1 / seconds, secondsPing, a, b), textOrg[6], fontFace, fontScale, color, 2, 8, false);

				cv::imshow("Esc for exit. 10/02/25", croppedImg);
			}
		}
		else {
			if (kSwitch > 0.1)
				kSwitch *= 0.8;
			transforms[1].dx *= 0.8;
			transforms[1].dy *= 0.8;
			transforms[1].da *= 0.8;
			transforms[1].getTransform(TStab, a, b, c, atan_ba, framePart);
			cuda::warpAffine(gFrame, gFrameStabilized, TStab, cv::Size(a, b));

			gFrameCrop = gFrameStabilized(roi);

			//cuda::cvtColor(gFrameCrop, gFrameCropHigh, COLOR_BGR2GRAY);
			//gFrameCropHigh.convertTo(gFrameCropHigh, CV_8UC1);
			////highPassFilterBGR->apply(gFrameCrop, gFrameCropHigh);
			//laplasianFilterGray->apply(gFrameCropHigh, gFrameCropHigh);
			//maxBoxFilterGray->apply(gFrameCropHigh, gFrameCropHigh);
			//cuda::cvtColor(gFrameCropHigh, gFrameCropHigh, COLOR_GRAY2BGR);
			//cuda::addWeighted(gFrameCrop, 1.0, gFrameCropHigh, 0.3, 0.8, gFrameCrop);



			cuda::resize(gFrameCrop, gFrameCrop, cv::Size(a, b), 0.0, 0.0, 3);
			gFrameCrop.download(croppedImg);
			//endPing = clock();
			if (writeVideo)
			{
				gFrame = gFrame(roi);
				cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, 3);
				gFrame.download(frame);

				//cv::putText(frame, format("Raw Video Stream Crop"), textOrg[0], fontFace, fontScale, color, 2, 8, false);
				frame.copyTo(writerFrame(cv::Rect(0, 0, frame.cols, frame.rows)));

				croppedImg.copyTo(writerFrame(cv::Rect(0, frame.rows, frame.cols, frame.rows)));


				cv::putText(writerFrame, format("3) Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), textOrg[0], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Wienner[1] %d, thread[t] %d, stab %d", wienner, threadwiener, stabPossible), textOrg[1], fontFace, fontScale, Scalar(0, 0, 255), 2, 8, false);
				cv::putText(writerFrame, format("[x y angle] %2.0f %2.0f %1.1f]", TStab.at<double>(0, 2), TStab.at<double>(1, 2), 180.0 / 3.14 * atan2(TStab.at<double>(1, 0), TStab.at<double>(0, 0))),
					textOrg[2], fontFace, fontScale, Scalar(0, 0, 255), 2, 8, false);
				cv::putText(writerFrame, format("[dx dy] %2.2f %2.2f]", d.x, d.y), textOrg[3], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Tau[3][4] = %3.0f, kSwitch = %1.2f", tauStab, kSwitch), textOrg[4], fontFace, fontScale, color, 2, 8, false);
				//cv::putText(writerFrame, format("Cut-off[w][s] %2.2f, %d Corners of %d.", 1 / framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Crop[w][s] = %2.1f ,%d Corners of %d.", 1 / framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("fps = %2.1f, ping = %1.3f", 1 / seconds, secondsPing), textOrg[6], fontFace, fontScale, color, 2, 8, false);


				//if (p0.size() > 0)
				//	for (uint i = 0; i < p0.size(); i++)
				//		circle(writerFrame, cv::Point2f(p1[i].x + a, p1[i].y), 8, colors[i], -1);

				cv::imshow("Writed", writerFrame);
				writer.write(writerFrame);

				writerSmall.write(croppedImg);

			}
			else {
				cv::putText(croppedImg, format("4) Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), textOrg[0], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("Wienner[1] %d, thread[t] %d", wienner, threadwiener), textOrg[1], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("[x y angle] %2.0f %2.0f %1.1f]", TStab.at<double>(0, 2), TStab.at<double>(1, 2), 180.0 / 3.14 * atan2(TStab.at<double>(1, 0), TStab.at<double>(0, 0))),
					textOrg[2], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("[dx dy] %2.2f %2.2f]", d.x, d.y), textOrg[3], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("Tau[3][4] = %3.0f, kSwitch = %1.2f", tauStab, kSwitch), textOrg[4], fontFace, fontScale, color, 2, 8, false);
				//cv::putText(croppedImg, format("Crop[w][s] %2.2f, %d Corners of %d.", 1/framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Crop[w][s] = %2.1f ,%d Corners of %d.", 1 / framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("fps = %2.1f, ping = %1.3f", 1 / seconds, secondsPing), textOrg[6], fontFace, fontScale, color, 2, 8, false);

				cv::imshow("Esc for exit. 10/02/25", croppedImg);
			}

		}

		// Ожидание внешних команд управления с клавиатуры

		int keyboard = waitKey(1);
		if (keyboard == 'c')
		{
			imwrite("imgInCam.jpg", frame);
			imwrite("imgOutCam.jpg", croppedImg);
		}
		if (keyboard == 'q' || keyboard == 27)
			break;
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
	}
	outputFile.close();
	capture.release();
	return 0;
}