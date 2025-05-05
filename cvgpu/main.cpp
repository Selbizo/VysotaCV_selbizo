//Алгоритм стабилизации видео на основе вычисление Lucas-Kanade Optical Flow

#include "testGpuFunctions.hpp"
#include "wienerFilter.hpp"

using namespace cv;
using namespace std;

int main()
{
	//string videoSource = "http://192.168.0.102:4747/video"; // pad6-100, pixel4-101, pixel-102
	//string videoSource = "http://10.108.144.71:4747/video"; // pad6-100, pixel4-101, pixel-102
	//string videoSource = "http://192.168.0.103:4747/video"; // pad6-100, pixel4-101, pixel-102
	string videoSource = "./SourceVideos/treeshd.mp4"; // pad6-100, pixel4-101, pixel-102
	//int videoSource = 0;

	bool writeVideo = true;
	bool stabPossible = false;

	// Создадим массив случайных цветов для цветов характерных точек
	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < 1000; i++)
	{
		int b = rng.uniform(88, 188);
		int g = rng.uniform(80, 130);
		int r = rng.uniform(165, 255);
		colors.push_back(Scalar(b, g, r));
	}
	// переменные для поиска характерных точек
	//int     = 2; //коэффициент сжатия изображения для обработки //пока не работает
	vector<uchar> status;
	//vector<Point2f> err;
	Mat err;

	int	srcType = CV_8UC1;
	int maxCorners = 400   ;      //100/n
	double qualityLevel = 0.003; //0.0001
	double minDistance = 6.0   + 2.0; //8.0
	int blockSize = 35   + 10; //45 80 максимальное значение окна
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
	vector <TransformParam> velocity(2);
	for (int i = 0; i < 2;i++)
	{
		velocity[i].dx = 0.0;
		velocity[i].dy = 0.0;
		velocity[i].da = 0.0;
	}

	//для гомографии

	// Траектория камеры
	vector<Mat> homoTransforms;


	// переменные для фильтра Виннера
	Mat Hw, h, frameGray_wiener;
	cuda::GpuMat gHw, gH, gFrameGrayWiener;

	bool wiener = false;
	bool threadwiener = false;
	double nsr = 0.01;
	double Q = 8.0; // скважность считывания кадра на камере (выдержка к частоте кадров) (умножена на 10)
	double LEN = 0;
	double THETA = 0.0;

	//для обработки трех каналов по Виннеру
	vector<Mat> channels(3), channelsWiener(3);
	Mat frame_wiener;

	vector<cuda::GpuMat> gChannels(3), gChannelsWiener(3);
	cuda::GpuMat gFrameWiener;

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~для вывода изображения на дисплей~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Mat croppedImg, frame_crop;
	cuda::GpuMat gCroppedImg, gFrameCrop, gFrameCropHigh;

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ для счетчика кадров в секунду ~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	int frameCnt = 0;
	double seconds = 0.0;
	double secondsPing = 0.0;
	clock_t start = clock();
	clock_t end = clock();

	clock_t startPing = clock();
	clock_t endPing = clock();

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Захват первого кадра ~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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

	const double a = oldFrame.cols  ;
	const double b = oldFrame.rows  ;
	const double c = sqrt(a * a + b * b);
	const double atan_ba = atan2(b, a);

	roi.x = a * ((1.0 - framePart) / 2.0);
	roi.y = b * ((1.0 - framePart) / 2.0);
	roi.width = a * framePart;
	roi.height = b * framePart;

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Для отображения надписей на кадре~~~~~~~~~~~~~~~~~~~~~~~~~~~
	setlocale(LC_ALL, "RU");
	vector <Point> textOrg(10);
	vector <Point> textOrgCrop(10);
	vector <Point> textOrgStab(10);
	vector <Point> textOrgOrig(10);
	if (writeVideo)
	{ 
		for (int i = 0; i < 10; i++)
		{
			textOrg[i].x = 5 + a;
			textOrg[i].y = 5 + 32 * (i + 1) + b;

			textOrgCrop[i].x = 5;
			textOrgCrop[i].y = 5 + 32 * (i + 1) + b;

			textOrgStab[i].x = 5;
			textOrgStab[i].y = 5 + 32 * (i + 1);

			textOrgOrig[i].x = 5 + a;
			textOrgOrig[i].y = 5 + 32 * (i + 1);
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
	double fontScale = 1.3;
	Scalar color(82, 156, 23);
	Scalar colorRED(48, 62, 255);
	Scalar colorGREEN(82, 156, 23);
	Scalar colorBLUE(239, 107, 23);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создадим маску для нахождения точек~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Mat mask_host = Mat::zeros(cv::Size(a, b), CV_8U);
	rectangle(mask_host, Rect(a * (1.0 - 0.8) / 2, b * (1.0 - 0.8) / 2, a * 0.8, b * 0.8), Scalar(255), FILLED); // Прямоугольная маска
	cuda::GpuMat mask_device(mask_host);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем GpuMat для мнимой части фильтра Винера~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cuda::GpuMat zeroMatH(cv::Size(a, b), CV_32F, Scalar(0)), complexH;
	
	Ptr<cuda::DFT> forwardDFT = cuda::createDFT(cv::Size(a, b), DFT_SCALE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> inverseDFT = cuda::createDFT(cv::Size(a, b), DFT_INVERSE | DFT_COMPLEX_INPUT);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем объект для записи отклонения в CSV файл~~~~~~~~~~~~~~~~~~~~~~~~~~~
	std::ofstream outputFile("./OutputResults/StabOutputs.txt");
	if (!outputFile.is_open())
	{
		cout << "Не удалось открыть файл для записи" << endl;
		return -1;
	}

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Запись заголовка в CSV файл

	outputFile << "FrameNumber\tdx\tdy\tX\tY\ttr2x\ttr2y\ttr3x\ttr3y" << endl;
	//outputFile << frameCount << "\t" << xDev << "\t" << yDev << "\t" << transforms[1].dx << "\t" << transforms[1].dy << transforms[2].dx << "\t" << transforms[2].dy << transforms[3].dx << "\t" << transforms[3].dy << endl;
	int frameCount = 0;

	while (true) {
		initFirstFrame(capture, oldFrame, gOldFrame, gOldGray, gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms,    kSwitch, a, b, mask_device, stabPossible);
		if (stabPossible)
			break;
	}
	
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~создание объекта записи видео
	VideoWriter writer, writerSmall;
	cv::Mat writerFrame(oldFrame.rows * 2, oldFrame.cols * 2, CV_8UC3), writerFrameSmall(oldFrame.rows, oldFrame.cols, CV_8UC3);
	cv::Mat writerFrameToShow;
	int selected_bitrate = 10'000'000;
	if (writeVideo) {
		bool isColor = (oldFrame.type() == CV_8UC3);
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZE VIDEOWRITER~~~~~~~~~~~~~~~~~~~~~~~~~~~

		//int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
		int codec = VideoWriter::fourcc('D', 'I', 'V', 'X');  // select desired codec (must be available at runtime)

		double fps = 30.0;                          // framerate of the created video stream
		string filename = "./OutputVideos/TestVideo.avi";             // name of the output video file
		string filenameSmall = "./OutputVideos/StabilizatedVideo.avi";  // name of the output video file

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
					p1[i].x     ;
					p1[i].y     ;
					p0.push_back(p1[i]); // Выбор точек good_new
				}
			}

			if (p1.size() < maxCorners * 5 / 7 && rng.uniform(0.0, 1.0) < 0.8)
			{
				//p0.push_back(Point2f(a / 2 + rng.uniform(-a / 2, a / 3), b / 2 + rng.uniform(-b/3, b/3)));
				for (uint i = 0; i < abs((int)(velocity[0].dx + velocity[0].dy))*2 + 1; ++i)
				{
					p0.push_back(Point2f(transforms[1].dx / 2 + a / 2 + rng.uniform(-a / 3, a / 3), transforms[1].dy / 2 + b / 2 + rng.uniform(-b / 3, b / 3)));
				}
			}

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

			if (wiener == false && p0.size() > 0 && 0)
			{
				for (uint i = 0; i < p0.size(); i++)
					circle(frame, p1[i], 4, colors[i], -1);
			}

			gFrame.upload(frame);

			cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
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

			cuda::cvtColor(gFrame, gFrameGray, COLOR_BGR2GRAY);
			cuda::resize(gFrameGray, gFrameGray, cv::Size(a   , b   ), 0.0, 0.0, INTER_CUBIC);

			if (frameCnt % 10 == 1 && !stabPossible)
			{
				initFirstFrame(capture, oldFrame, gOldFrame, gOldGray, gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms,    kSwitch, a, b, mask_device, stabPossible); //70ms
				//stabPossible = 0;
			}
			else if (frameCnt % 50 == 1 && !stabPossible)
			{

			}
			else
				initFirstFrameZero(oldFrame, gOldFrame, gOldGray, gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms,    kSwitch, a, b, mask_device, stabPossible); //70ms

			if (stabPossible) {
				d_pyrLK_sparse->calc(gOldGray, gFrameGray, gP0, gP1, gStatus, gErr);
				gP1.download(p1);
				gErr.download(err);
			}
			//cout << "err.Size() = " << err.Size();
		}
		else if (stabPossible) {
			cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
			cuda::resize(gFrameGray, gFrameGray, cv::Size(a   , b   ), 0.0, 0.0, INTER_CUBIC);
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
			getBiasAndRotation(p0, p1, d, transforms, T); //уже можно делать Винеровскую фильтрацию

			



			iirAdaptive(transforms, tauStab, roi, gFrame.cols, gFrame.rows, kSwitch, velocity);
			transforms[1].getTransform(TStab, a, b, c, atan_ba, framePart);

			if (T.rows == 2 && T.cols == 3)
			{
				double xDev = T.at<double>(0, 2);
				double yDev = T.at<double>(1, 2);
				// Запись отклонения в CSV файл
				outputFile << frameCount << "\t" << xDev << "\t" << yDev << "\t" << transforms[1].dx <<"\t" << transforms[1].dy << "\t" << transforms[2].dx << "\t" << transforms[2].dy << "\t" << transforms[3].dx << "\t" << transforms[3].dy << endl;
				//outputFile << frameCount << "," << round(xDev * 100.0) / 100.0 << "," << round(yDev * 100.0) / 100.0 << "," << round(transforms[1].dx * 100.0) / 100.0 << "," << round(transforms[1].dy * 100.0) / 100.0 << endl;
			}
			

			// Винеровская фильтрация
			if (wiener && kSwitch > 0.01)
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
						Gfilter2DFreqV2(gChannels[i], gChannelsWiener[i], complexH, forwardDFT, inverseDFT);
					}
				}
				else
				{
					std::thread blueChannelWiener(channelWiener, &gChannels[0], &gChannelsWiener[0], &complexH, &forwardDFT, &inverseDFT);
					std::thread greenChannelWiener(channelWiener, &gChannels[1], &gChannelsWiener[1], &complexH, &forwardDFT, &inverseDFT);
					std::thread redChannelWiener(channelWiener, &gChannels[2], &gChannelsWiener[2], &complexH, &forwardDFT, &inverseDFT);

					blueChannelWiener.join();
					greenChannelWiener.join();
					redChannelWiener.join();
				}
				cuda::merge(gChannelsWiener, gFrame);
				gFrame.convertTo(gFrame, CV_8UC3);
				cuda::bilateralFilter(gFrame, gFrame, 13, 5.0, 3.0);
			}

			cuda::warpAffine(gFrame, gFrameStabilized, TStab, cv::Size(a, b));

			gFrameCrop = gFrameStabilized(roi); 

			cuda::resize(gFrameCrop, gFrameCrop, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
			gFrameCrop.download(croppedImg);
			endPing = clock();

			
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~ Вывод изображения на дисплей
			if (writeVideo)
			{
				gFrame = gFrame(roi);
				cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
				gFrame.download(frame);
				
				frame.copyTo(writerFrame(cv::Rect(0, frame.rows, frame.cols, frame.rows)));
				croppedImg.copyTo(writerFrame(cv::Rect(0, 0, frame.cols, frame.rows)));

				if (p0.size() > 0)
					for (uint i = 0; i < p0.size(); i++)
						circle(writerFrame, cv::Point2f(p1[i].x + a, p1[i].y), 4, colors[i], -1);
								
				cv::ellipse(writerFrame, cv::Point2f(-transforms[1].dx + a + a / 2, -transforms[1].dy + b / 2), cv::Size(a * framePart / 2, 0), 0.0 - 0.0*transforms[1].da * RAD_TO_DEG, 0, 360, Scalar(20, 200, 10), 2);
				cv::ellipse(writerFrame, cv::Point2f(-transforms[1].dx + a + a / 2, -transforms[1].dy + b / 2), cv::Size(0, b * framePart / 2), 0.0 - 0.0*transforms[1].da * RAD_TO_DEG, 0, 360, Scalar(20, 200, 10), 2);

				cv::putText(writerFrame, format("It's OK. WnrFltr Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), 
					textOrg[0], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Wnr On[1] %d, threads On[t] %d, stab On %d", wiener, threadwiener, stabPossible), 
					textOrg[1], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("[X Y Roll] %2.2f %2.2f %2.2f]", transforms[1].dx, transforms[1].dy, transforms[1].da*RAD_TO_DEG), 
					textOrg[2], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("[dX dY dRoll] %2.2f %2.2f %2.2f]", transforms[0].dx, transforms[0].dy, transforms[0].da), 
					textOrg[3], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("[skoX skoY skoRoll] %2.2f %2.2f %2.2f]", transforms[3].dx, transforms[3].dy, transforms[3].da), 
					textOrg[4], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("[vX vY vRoll] %2.2f %2.2f %2.2f]", velocity[0].dx, velocity[0].dy, velocity[0].da), 
					textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Filter time[3][4]= %3.0f frames, filter power = %1.2f", tauStab, kSwitch), 
					textOrg[6], fontFace, fontScale/1.2, color, 2, 8, false);
				cv::putText(writerFrame, format("Crop[w][s] = %2.2f, %d Current corners of %d.", 1 / framePart, gP0.cols, maxCorners), 
					textOrg[7], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("FPS = %2.1f, Ping = %1.3f.", 1 / seconds, secondsPing), 
					textOrg[8], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Image resolution: %3.0f x %3.0f.", a, b), 
					textOrg[9], fontFace, fontScale / 1.2, color, 2, 8, false);
				cv::putText(writerFrame, format("Original video"), textOrgOrig[0], fontFace, fontScale * 1.3, color, 2, 8, false);
				cv::putText(writerFrame, format("Without Stab"), textOrgCrop[0], fontFace, fontScale * 1.3, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("Stab ON"), textOrgStab[0], fontFace, fontScale * 1.3, color, 2, 8, false);
								
				writer.write(writerFrame);
				writerSmall.write(croppedImg);
				cv::resize(writerFrame, writerFrameToShow, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
				cv::imshow("Writed", writerFrameToShow);
			}
			else {
				cv::putText(croppedImg, format("No recording Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), textOrg[0], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("Wiener[1] %d, thread[t] %d", wiener, threadwiener), textOrg[1], fontFace, fontScale, color, 2, 8, false);
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

			cuda::resize(gFrameCrop, gFrameCrop, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
			gFrameCrop.download(croppedImg);
			//endPing = clock();
			if (writeVideo)
			{
				gFrame = gFrame(roi);
				cuda::resize(gFrame, gFrame, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
				gFrame.download(frame);

				frame.copyTo(writerFrame(cv::Rect(0, 0, frame.cols, frame.rows)));

				croppedImg.copyTo(writerFrame(cv::Rect(0, frame.rows, frame.cols, frame.rows)));



				cv::putText(writerFrame, format("NOT GOOD. WnrFltr Q[5][6] = % 2.1f, SNR[7][8] = % 2.1f", Q, 1 / nsr),
					textOrg[0], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("Wnr On[1] %d, threads On[t] %d, stab On %d", wiener, threadwiener, stabPossible),
					textOrg[1], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("[X Y Roll] %2.2f %2.2f %2.2f]", transforms[1].dx, transforms[1].dy, transforms[1].da* RAD_TO_DEG),
					textOrg[2], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("[dX dY dRoll] %2.2f %2.2f %2.2f]", transforms[0].dx, transforms[0].dy, transforms[0].da),
					textOrg[3], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("[skoX skoY skoRoll] %2.2f %2.2f %2.2f]", transforms[3].dx, transforms[3].dy, transforms[3].da),
					textOrg[4], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("[vX vY vRoll] %2.2f %2.2f %2.2f]", velocity[0].dx, velocity[0].dy, velocity[0].da),
					textOrg[5], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("Filter time[3][4]= %3.0f frames, filter power = %1.2f", tauStab, kSwitch),
					textOrg[6], fontFace, fontScale / 1.2, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("Crop[w][s] = %2.2f, %d Current corners of %d.", 1 / framePart, gP0.cols, maxCorners),
					textOrg[7], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("FPS = %2.1f, Ping = %1.3f.", 1 / seconds, secondsPing),
					textOrg[8], fontFace, fontScale, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("Image resolution: %3.0f x %3.0f.", a, b),
					textOrg[9], fontFace, fontScale / 1.2, colorRED, 2, 8, false);

				cv::putText(writerFrame, format("Original video"), textOrgOrig[0], fontFace, fontScale * 1.3, color, 2, 8, false);
				cv::putText(writerFrame, format("Without Stab"), textOrgCrop[0], fontFace, fontScale * 1.3, colorRED, 2, 8, false);
				cv::putText(writerFrame, format("Stab error"), textOrgStab[0], fontFace, fontScale * 1.3, colorRED, 2, 8, false);

				writer.write(writerFrame);
				writerSmall.write(croppedImg);
				cv::resize(writerFrame, writerFrameToShow, cv::Size(a, b), 0.0, 0.0, INTER_CUBIC);
				cv::imshow("Writed", writerFrameToShow);

			}
			else {
				cv::putText(croppedImg, format("No recording Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr), textOrg[0], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("Wiener[1] %d, thread[t] %d", wiener, threadwiener), textOrg[1], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("[x y angle] %2.0f %2.0f %1.1f]", TStab.at<double>(0, 2), TStab.at<double>(1, 2), 180.0 / 3.14 * atan2(TStab.at<double>(1, 0), TStab.at<double>(0, 0))),
					textOrg[2], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("[dx dy] %2.2f %2.2f]", d.x, d.y), textOrg[3], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("Tau[3][4] = %3.0f, kSwitch = %1.2f", tauStab, kSwitch), textOrg[4], fontFace, fontScale, color, 2, 8, false);
				cv::putText(writerFrame, format("Crop[w][s] = %2.1f ,%d Corners of %d.", 1 / framePart, gP0.cols, maxCorners), textOrg[5], fontFace, fontScale, color, 2, 8, false);
				cv::putText(croppedImg, format("fps = %2.1f, ping = %1.3f", 1 / seconds, secondsPing), textOrg[6], fontFace, fontScale, color, 2, 8, false);

				cv::imshow("Esc for exit. 10/02/25", croppedImg);
			}

		}

		// Ожидание внешних команд управления с клавиатуры

		int keyboard = waitKey(1);
		if (keyResponse(keyboard, frame, croppedImg, a, b, nsr, wiener, threadwiener, Q, tauStab, framePart, roi))
			break;
		
	}
	outputFile.close();
	capture.release();
	return 0;
}

//#include <opencv2/opencv.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudafeatures2d.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <chrono>
//
//using namespace cv;
//using namespace std;
//using namespace std::chrono;
//
//class DroneSpeedEstimator {
//private:
//    // CUDA объекты
//    Ptr<cuda::ORB> orb;
//    Ptr<cuda::DescriptorMatcher> matcher;
//    cuda::GpuMat prevFrame, prevGray;
//    vector<Point2f> prevPoints;
//    double focalLength;  // Фокусное расстояние в пикселях
//    double altitude;     // Высота в метрах
//    double scaleFactor;  // Масштабный коэффициент (пиксели/метр)
//    high_resolution_clock::time_point prevTime;
//
//public:
//    DroneSpeedEstimator(double focal, double alt)
//        : focalLength(focal), altitude(alt) {
//        // Инициализация детектора ORB на GPU
//        orb = cuda::ORB::create(1000);
//        matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
//
//        // Расчет масштабного коэффициента
//        scaleFactor = focalLength / altitude;
//        prevTime = high_resolution_clock::now();
//    }
//
//    double estimateSpeed(const Mat& frame) {
//        auto currTime = high_resolution_clock::now();
//        double timeDelta = duration_cast<milliseconds>(currTime - prevTime).count() / 1000.0;
//        prevTime = currTime;
//
//        cuda::GpuMat d_frame, d_gray, d_keypoints, d_descriptors;
//        vector<KeyPoint> keypoints;
//        Mat descriptors;
//
//        // Загрузка кадра на GPU
//        d_frame.upload(frame);
//        cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);
//
//        // Первый кадр - инициализация
//        if (prevPoints.empty()) {
//            orb->detectAndComputeAsync(d_gray, noArray(), d_keypoints, d_descriptors);
//            orb->convert(d_keypoints, keypoints);
//            KeyPoint::convert(keypoints, prevPoints);
//            prevGray = d_gray.clone();
//            return 0.0;
//        }
//
//        // Находим ключевые точки в текущем кадре
//        orb->detectAndComputeAsync(d_gray, noArray(), d_keypoints, d_descriptors);
//        orb->convert(d_keypoints, keypoints);
//
//        vector<Point2f> currPoints;
//        KeyPoint::convert(keypoints, currPoints);
//
//        // Находим соответствия точек
//        vector<DMatch> matches;
//        matcher->match(d_descriptors, matches);
//
//        // Фильтрация хороших соответствий
//        vector<Point2f> goodPrev, goodCurr;
//        for (size_t i = 0; i < matches.size(); i++) {
//            if (matches[i].distance < 25.0) {
//                goodPrev.push_back(prevPoints[matches[i].queryIdx]);
//                goodCurr.push_back(currPoints[matches[i].trainIdx]);
//            }
//        }
//
//        // Рассчитываем среднее смещение точек
//        double totalDisplacement = 0.0;
//        int validPoints = 0;
//
//        for (size_t i = 0; i < goodPrev.size(); i++) {
//            double dx = goodCurr[i].x - goodPrev[i].x;
//            double dy = goodCurr[i].y - goodPrev[i].y;
//            totalDisplacement += sqrt(dx * dx + dy * dy);
//            validPoints++;
//        }
//
//        if (validPoints == 0) return 0.0;
//
//        // Среднее смещение в пикселях
//        double meanDisplacement = totalDisplacement / validPoints;
//
//        // Переводим смещение в метры и вычисляем скорость
//        double groundDisplacement = meanDisplacement / scaleFactor; // в метрах
//        double speed = groundDisplacement / timeDelta;             // м/с
//
//        // Обновляем данные для следующего кадра
//        prevGray = d_gray.clone();
//        prevPoints = currPoints;
//
//        return speed;
//    }
//};
//
//int main() {
//    // Параметры камеры (пример для DJI Phantom 4 Pro)
//    double focalLengthPx = 3820;  // Фокусное расстояние в пикселях
//    double altitude = 50.0;       // Высота полета в метрах
//
//    DroneSpeedEstimator speedEstimator(focalLengthPx, altitude);
//
//    VideoCapture cap(0);
//    if (!cap.isOpened()) {
//        cerr << "Error opening video file" << endl;
//        return -1;
//    }
//
//    Mat frame;
//    while (cap.read(frame)) {
//        double speed = speedEstimator.estimateSpeed(frame);
//        /*cout << "Estimated speed: " << speed << " m/s ("
//            << speed * 3.6 << " km/h)" << endl;*/
//
//        putText(frame, format("Speed: %.2f m/s", speed),
//            Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
//        imshow("Drone View", frame);
//        if (waitKey(1) == 27) break;
//    }
//
//    return 0;
//}