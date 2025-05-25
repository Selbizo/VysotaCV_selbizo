//
//#include <opencv2/core.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
//
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//
//#include <iostream>
//#include <stdio.h>
//
//#include "Config.hpp"
//#include "basicFunctions.hpp"
//#include "stabilizationFunctions.hpp"
//
//using namespace cv;
//using namespace std;
//
//
//
//
//int main(int, char**)
//{
//    Mat src, out, TStab(2, 3, CV_64F);
//    TransformParam noise = { 0.0, 0.0, 0.0 };
//    RNG rng;
//
//    // use default camera as video source
//    VideoCapture cap(videoSource);
//    //VideoCapture cap(0);
//    // check if we succeeded
//    if (!cap.isOpened()) {
//        cerr << "ERROR! Unable to open camera\n";
//        return -1;
//    }
//    // get one frame from camera to know frame size and type
//    cap >> src;
//    // check if we succeeded
//    if (src.empty()) {
//        cerr << "ERROR! blank frame grabbed\n";
//        return -1;
//    }
//    bool isColor = (src.type() == CV_8UC3);
//    Rect roi;
//
//    roi.x = src.cols * 1 / 8;
//    roi.y = src.rows * 1 / 8;
//    roi.width = src.cols * 3 / 4;
//    roi.height = src.rows * 3 / 4;
//
//    //--- INITIALIZE VIDEOWRITER
//    VideoWriter writer;
//    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
//
//    double fps = 30.0;
//    string filename = "./OutputVideos/ShakedVideo.avi";
//
//    writer.open(filename, codec, fps, roi.size(), isColor);
//
//    // check if we succeeded
//    if (!writer.isOpened()) {
//        cerr << "Could not open the output video file for write\n";
//        return -1;
//    }
//
//    //--- GRAB AND WRITE LOOP
//    cout << "Writing videofile: " << filename << endl
//        << "Press any key to terminate" << endl;
//
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем объект для записи отклонения в CSV файл~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    std::ofstream outputFile("./OutputResults/StabOutputs.txt");
//    if (!outputFile.is_open())
//    {
//        cout << "Не удалось открыть файл для записи" << endl;
//        return -1;
//    }
//
//
//    for (;;)
//    {
//        cap >> src;
//        // check if we succeeded
//        if (src.empty()) {
//            cerr << "Ending.\n";
//            break;
//        }
//        noise.dx = (double)(rng.uniform(-100.0, 100.0)) / 32 + noise.dx * 31 / 32;
//        noise.dy = (double)(rng.uniform(-100.0, 100.0)) / 32 + noise.dy * 31 / 32;
//        noise.da = (double)(rng.uniform(-1000.0, 1000.0) * 0.0001) / 32 + noise.da * 31 / 32;
//
//        noise.getTransform(TStab);
//        warpAffine(src, out, TStab, src.size());
//
//        out = out(roi);
//        // encode the frame into the videofile stream
//        writer.write(out);
//        // show live and wait for a key with timeout long enough to show images
//        imshow("Live", out);
//        if (waitKey(1) >= 0)
//            break;
//    }
//    // the videofile will be closed and released automatically in VideoWriter destructor
//    return 0;
//}

//#include "Config.hpp"
//#include "basicFunctions.hpp"
//#include "stabilizationFunctions.hpp"
//
//int main() {
//    // System dimensions
//    int state_dim = 4;  // vx, vy, ax, ay
//    int meas_dim = 2;   // vx, vy
//
//    // Create system matrices
//    double FPS = 30.0;
//    double dt = 1;//1 / FPS;
//    cv::Mat A = (cv::Mat_<double>(4, 4) <<
//        1, 0, dt, 0,
//        0, 1, 0, dt,
//        0, 0, 1, 0,
//        0, 0, 0, 1);
//
//    cv::Mat C = (cv::Mat_<double>(2, 4) <<
//        1, 0, 0, 0,
//        0, 1, 0, 0);
//
//    cv::Mat Q = cv::Mat::eye(4, 4, CV_64F) * 0.01;
//    cv::Mat R = cv::Mat::eye(2, 2, CV_64F) * 0.1;
//    cv::Mat P = cv::Mat::eye(4, 4, CV_64F) * 1;
//
//    // Create Kalman filter
//    KalmanFilterCV kf(dt, A, C, Q, R, P);
//
//    // Initialize with first measurement
//    cv::Mat x0 = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
//    kf.init(0, x0);
//
//    // Simulate measurements and update
//    for (int i = 0; i < 40; i++) {
//        cv::Mat measurement = (cv::Mat_<double>(2, 1) << i * 1.0, i * 0.5);
//        std::cout << "measurement: " << measurement.t() << std::endl;
//        kf.update(measurement);
//
//        cv::Mat state = kf.state();
//        std::cout << "State: " << state.t() << std::endl;
//    }
//
//    return 0;
//}