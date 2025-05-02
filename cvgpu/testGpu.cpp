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

    int ndisp = 64;
    int blocksize = 31;

    Ptr<cuda::StereoBM> bm;
    Ptr<cuda::StereoSGM> sgm;
    bm = cuda::createStereoBM(ndisp, blocksize);
    

    sgm = cuda::createStereoSGM(0, ndisp);


    cv::VideoCapture cap_right, cap_left;
    bool pixelLeft = true;
    // Открываем две камеры (указываем соответствующие индексы)
    //if (pixelLeft)
    //{
    ////cv::VideoCapture cap_left("http://10.142.69.133:4747/video"); //pixel
    ////cv::VideoCapture cap_right("http://10.142.69.71:4747/video"); //xiaomi

    //cap_left.open("http://192.168.48.133:4747/video"); //pixel 1
    //cap_right.open("http://192.168.48.136:4747/video"); //pixel 4
    //}
    //else
    //{
    //cap_left.open("http://192.168.48.136:4747/video"); //pixel 4
    //cap_right.open("http://192.168.48.133:4747/video"); //pixel 1
    //}


    cap_left.open(0); //pixel 4
    cap_right.open(1); //pixel 1

    //if (pixelLeft)
    //{
    //    //cv::VideoCapture cap_left("http://10.142.69.133:4747/video"); //pixel
    //    //cv::VideoCapture cap_right("http://10.142.69.71:4747/video"); //xiaomi

    //    cap_left.open("http://10.142.69.133:4747/video"); //pixel
    //    cap_right.open("http://10.142.69.71:4747/video"); //xiaomi
    //}
    //else
    //{
    //    cap_left.open("http://10.142.69.71:4747/video"); //xiaomi
    //    cap_right.open("http://10.142.69.133:4747/video"); //pixel
    //}

    if (!cap_left.isOpened() || !cap_right.isOpened()) {
        std::cerr << "Ошибка открытия камер!" << std::endl;
        return -1;
    }

    // Устанавливаем разрешение (желательно синхронизировать для обеих камер)
    cap_left.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap_left.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap_right.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap_right.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Prepare disparity map of specified type
    Mat disp_bm(left.size(), CV_8U);
    Mat disp_sgm(left.size(), CV_8U);
    cuda::GpuMat d_disp_bm(left.size(), CV_8U);
    cuda::GpuMat d_disp_sgm(left.size(), CV_8U);

    cout << endl;

    // Параметры камеры
    double f = 100.0;          // Фокусное расстояние (пиксели)
    double cx = 320.0, cy = 240.0; // Оптический центр
    double B = 0.135;            // База (13.5 см)

    // Матрица Q
    cv::Mat Q = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, -cx,
        0, 1, 0, -cy,
        0, 0, 0, f,
        0, 0, -1 / B, 0);

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


        cvtColor(left_src, left, COLOR_BGR2GRAY);
        cvtColor(right_src, right, COLOR_BGR2GRAY);


        GaussianBlur(left, left, Size(3, 3), 5.0, 5.0);
        GaussianBlur(right, right, Size(3, 3), 5.0, 5.0);
        cv::normalize(left, left, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::normalize(right, right, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        //cv::equalizeHist(left, left);
        //cv::equalizeHist(right, right);

        d_left.upload(left);
        d_right.upload(right);

        /*cv::cuda::equalizeHist(d_left, d_left);
        cv::cuda::equalizeHist(d_right, d_right);*/

        imshow("left", left);
        imshow("right", right);

        bm->compute(d_left, d_right, d_disp_bm);
        //sgm->compute(d_left, d_right, d_disp_sgm);

        // Show results
        d_disp_bm.download(disp_bm);
        //d_disp_sgm.download(disp_sgm);

                // Нормализация для визуализации
        cv::normalize(disp_bm, disp_bm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        // Преобразование диспаратности в глубину (Z = B*f / d)
        cv::Mat depth;
        cv::reprojectImageTo3D(disp_bm, depth, Q, true);



        imshow("disparity_bm", (Mat_<uchar>)disp_bm);
        //imshow("disparity_sgm", (Mat_<uchar>)disp_sgm);

        //imshow("disparity_bm", disp_bm);
        //imshow("disparity_sgm", disp_sgm);

        cv::imshow("Depth", depth);

        int key = waitKey(1);
        if (key == 27)
            break;
    }
    cap_left.release();
    cap_right.release();

    destroyAllWindows();
    return 0;
}
