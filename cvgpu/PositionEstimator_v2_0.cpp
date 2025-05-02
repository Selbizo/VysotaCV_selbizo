//
//
//#include "testGpuFunctions.hpp"
//
//
//int notmain() {
//
//    	//��� ����������� �������� �� �����
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
//    // ��������� ������ (������ ���� �������� ��� �����������)
//    double focalLength = 1000.0;  // � ��������
//    double cx = 640.0 / 2;        // ����� ����������� �� x
//    double cy = 480.0 / 2;        // ����� ����������� �� y
//
//    // ������������� ����������� (����� ������ � �������, ����� �������� �� ���������)
//    cv::VideoCapture cap(0);
//    if (!cap.isOpened()) {
//        std::cerr << "������ �������� ������!" << std::endl;
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
//        // ������ ����������
//        auto angles = estimateUAVOrientation(currFrame, prevFrame, focalLength, cx, cy);
//
//        // ����� �����������
//        //std::cout << "Roll: " << angles.roll * RAD_TO_DEG << "  "
//            //<< "Pitch: " << angles.pitch * RAD_TO_DEG << "  "
//            //<< "Yaw: " << angles.yaw * RAD_TO_DEG << " " << std::endl;
//
//        // ��������� ���������� ����
//        currFrame.copyTo(prevFrame);
//        cv::putText(frame, format("Roll= %2.2f, Pitch= %2.2f, Yaw= %2.2f", angles.roll * RAD_TO_DEG, angles.pitch * RAD_TO_DEG, angles.yaw * RAD_TO_DEG), textOrg[0], fontFace, fontScale, Scalar(120, 60, 255), 2, 8, false);
//        // �����������
//        cv::imshow("UAV View", frame);
//        if (cv::waitKey(30) >= 0) break;
//    }
//
//    return 0;
//}