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