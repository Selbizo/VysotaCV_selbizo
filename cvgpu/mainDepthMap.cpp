#include "opencv2/cudastereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    bool running;
    Mat left_src, right_src;
    Mat left, right, disp_bm;
    cuda::GpuMat d_left, d_right, d_disp_bm;

    int ndisp = 128;
    int blocksize = 13;

    Ptr<cuda::StereoBM> bm;

    bm = cuda::createStereoBM(ndisp, blocksize);
    left_src = cv::imread("./SourceImages/imgleft.jpg", IMREAD_COLOR);
    right_src = cv::imread("./SourceImages/imgright.jpg", IMREAD_COLOR);
        
    running = true;
    while (running)
    {
        if (left_src.empty() || right_src.empty()) {
            std::cerr << "Пустой кадр!" << std::endl;
            continue;
        }

        cvtColor(left_src, left, COLOR_BGR2GRAY);
        cvtColor(right_src, right, COLOR_BGR2GRAY);

        d_left.upload(left);
        d_right.upload(right);
        bm->compute(d_left, d_right, d_disp_bm);
        d_disp_bm.download(disp_bm);
        cv::normalize(disp_bm, disp_bm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        
        imshow("left", left);
        imshow("right", right);
        imshow("disparity_bm", (Mat_<uchar>)disp_bm);

        int key = waitKey(1);
        if (key == 27)
            break;
    }
    destroyAllWindows();
    return 0;
}
