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

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <vector>

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

using namespace cv;
using namespace std;

class VideoStabilizer {
private:
    // ��������� ������������
    const int SMOOTHING_RADIUS = 400; // ������ ���� �����������
    const int MAX_FEATURES = 500;    // ����. ���������� ����� ��� ������������
    const float MIN_MATCH_DISTANCE = 1.7f;

    // CUDA �������
    Ptr<cuda::ORB> orb;
    Ptr<cuda::DescriptorMatcher> matcher;
    cuda::GpuMat prevFrame, prevGray;
    vector<Point2f> prevPoints;

    // ���������� ������
    vector<Mat> transforms;

public:
    VideoStabilizer() {
        orb = cuda::ORB::create(MAX_FEATURES);
        matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
    }

    Mat stabilizeFrame(const Mat& frame) {
        cuda::GpuMat d_frame, d_gray, d_keypoints, d_descriptors;
        vector<KeyPoint> keypoints;
        Mat descriptors;

        // �������� ����� �� GPU
        d_frame.upload(frame);
        cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);

        // ������ ���� - �������������
        if (prevPoints.empty()) {
            orb->detectAndComputeAsync(d_gray, noArray(), d_keypoints, d_descriptors);
            orb->convert(d_keypoints, keypoints);
            KeyPoint::convert(keypoints, prevPoints);
            prevGray = d_gray.clone();
            return frame;
        }

        // ������� �������� ����� � ������� �����
        orb->detectAndComputeAsync(d_gray, noArray(), d_keypoints, d_descriptors);
        orb->convert(d_keypoints, keypoints);

        vector<Point2f> currPoints;
        KeyPoint::convert(keypoints, currPoints);

        // ������� ������������ �����
        vector<DMatch> matches;
        matcher->match(d_descriptors, matches);

        // ���������� ������� ������������
        vector<Point2f> goodPrev, goodCurr;
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i].distance < MIN_MATCH_DISTANCE) {
                goodPrev.push_back(prevPoints[matches[i].queryIdx]);
                goodCurr.push_back(currPoints[matches[i].trainIdx]);
            }
        }

        // ��������� ����������
        Mat H;
        if (goodPrev.size() >= 4) {
            H = findHomography(goodCurr, goodPrev, RANSAC);
            transforms.push_back(H);
        }
        else {
            transforms.push_back(Mat::eye(3, 3, CV_64F));
        }

        // ����������� ��������������
        Mat smoothH = smoothTransforms();

        // ��������� �������������� � �����
        cuda::GpuMat d_stabilized;
        cuda::warpPerspective(d_frame, d_stabilized, smoothH, frame.size(),
            INTER_LINEAR, BORDER_REPLICATE);

        // ��������� ������ ��� ���������� �����
        prevGray = d_gray.clone();
        prevPoints = currPoints;

        // ��������� ��������� � GPU
        Mat stabilized;
        d_stabilized.download(stabilized);
        return stabilized;
    }

private:
    Mat smoothTransforms() {
        if (transforms.size() < SMOOTHING_RADIUS) {
            return transforms.back();
        }

        // ��������� ������� �������������� � ����
        Mat sumH = Mat::zeros(3, 3, CV_64F);
        int count = 0;
        for (int i = transforms.size() - SMOOTHING_RADIUS; i < transforms.size(); i++) {
            sumH += transforms[i];
            count++;
        }
        return sumH / count;
    }
};

int NOTmain() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    VideoStabilizer stabilizer;
    VideoWriter writer("stabilized_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH),
            cap.get(CAP_PROP_FRAME_HEIGHT)));

    Mat frame;
    while (cap.read(frame)) {
        Mat stabilized = stabilizer.stabilizeFrame(frame);
        writer.write(stabilized);
        imshow("Stabilized", stabilized);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    return 0;
}