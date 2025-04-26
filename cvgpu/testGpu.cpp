#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/utility.hpp>
#include "opencv2/cudastereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    bool running;
    Mat left_src, right_src;
    Mat left, right;
    cuda::GpuMat d_left, d_right;

    int ndisp = 88;
    int blocksize = 31;

    Ptr<cuda::StereoBM> bm;
    bm = cuda::createStereoBM(ndisp, blocksize);
    // Открываем две камеры (указываем соответствующие индексы)

    cv::VideoCapture cap_right("http://10.142.69.71:4747/video");
    cv::VideoCapture cap_left(0);

    if (!cap_left.isOpened() || !cap_right.isOpened()) {
        std::cerr << "Ошибка открытия камер!" << std::endl;
        return -1;
    }

    // Устанавливаем разрешение (желательно синхронизировать для обеих камер)
    cap_left.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_left.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap_right.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_right.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Prepare disparity map of specified type
    Mat disp(left.size(), CV_8U);
    cuda::GpuMat d_disp(left.size(), CV_8U);

    cout << endl;


    running = true;
    while (running)
    {

        // Захватываем кадры с обеих камер
        cap_left >> left_src;
        cap_right >> right_src;

        if (left_src.empty() || right_src.empty()) {
            std::cerr << "Пустой кадр!" << std::endl;
            continue;
        }
        GaussianBlur(left_src, left_src, Size(9, 9), 9.0, 9.0);
        GaussianBlur(right_src, right_src, Size(9, 9), 9.0, 9.0);

        cvtColor(left_src, left, COLOR_BGR2GRAY);
        cvtColor(right_src, right, COLOR_BGR2GRAY);

        cv::equalizeHist(left, left);
        cv::equalizeHist(right, right);

        d_left.upload(left);
        d_right.upload(right);

        /*cv::cuda::equalizeHist(d_left, d_left);
        cv::cuda::equalizeHist(d_right, d_right);*/

        imshow("left", left);
        imshow("right", right);

        bm->compute(d_left, d_right, d_disp);

        // Show results
        d_disp.download(disp);

        imshow("disparity", (Mat_<uchar>)disp);

        int key = waitKey(1);
        if (key == 27)
            break;
    }
    cap_left.release();
    cap_right.release();

    destroyAllWindows();
    return 0;
}