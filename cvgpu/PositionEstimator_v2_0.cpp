//
//
//#include "testGpuFunctions.hpp"
//
//
//int notmain() {
//
//    	//Для отображения надписей на кадре
//	setlocale(LC_ALL, "RU");
//	vector <Point> textOrg(10);
//
//		for (int i = 0; i < 10; i++)
//		{
//			textOrg[i].x = 20;
//			textOrg[i].y = 20 + 30 * (i + 1);
//		}
//	
//
//
//	int fontFace = FONT_HERSHEY_PLAIN;
//	double fontScale = 1.2;
//	Scalar color(40, 140, 0);
//    // Параметры камеры (должны быть известны или калиброваны)
//    double focalLength = 1000.0;  // в пикселях
//    double cx = 640.0 / 2;        // центр изображения по x
//    double cy = 480.0 / 2;        // центр изображения по y
//
//    // Инициализация видеопотока (здесь пример с камерой, можно заменить на видеофайл)
//    cv::VideoCapture cap(0);
//    if (!cap.isOpened()) {
//        std::cerr << "Ошибка открытия камеры!" << std::endl;
//        return -1;
//    }
//
//    cv::Mat frame;
//    cap >> frame;
//    cv::cuda::GpuMat prevFrame, currFrame;
//    prevFrame.upload(frame);
//
//    while (true) {
//        cap >> frame;
//        if (frame.empty()) break;
//
//        currFrame.upload(frame);
//
//        // Оценка ориентации
//        auto angles = estimateUAVOrientation(currFrame, prevFrame, focalLength, cx, cy);
//
//        // Вывод результатов
//        //std::cout << "Roll: " << angles.roll * RAD_TO_DEG << "  "
//            //<< "Pitch: " << angles.pitch * RAD_TO_DEG << "  "
//            //<< "Yaw: " << angles.yaw * RAD_TO_DEG << " " << std::endl;
//
//        // Обновляем предыдущий кадр
//        currFrame.copyTo(prevFrame);
//        cv::putText(frame, format("Roll= %2.2f, Pitch= %2.2f, Yaw= %2.2f", angles.roll * RAD_TO_DEG, angles.pitch * RAD_TO_DEG, angles.yaw * RAD_TO_DEG), textOrg[0], fontFace, fontScale, Scalar(120, 60, 255), 2, 8, false);
//        // Отображение
//        cv::imshow("UAV View", frame);
//        if (cv::waitKey(30) >= 0) break;
//    }
//
//    return 0;
//}