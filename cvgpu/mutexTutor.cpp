//#include <opencv2/opencv.hpp>
//#include <opencv2/cudawarping.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <iostream>
//#include <thread>
//#include <mutex>
//
//// ���������� ��������� ��� �������� ����� � ��������
//struct FrameData {
//    cv::Mat frame_cpu;
//    cv::cuda::GpuMat frame_gpu;
//    std::mutex mtx;
//};
//
//// ������� ��� ��������� ������ �� GPU
//void processFrame(FrameData* data)
//{
//    // ������ ������� � ������
//    std::unique_lock<std::mutex> lock(data->mtx);
//
//    // ������������ ���� � ������� ������
//    cv::cuda::cvtColor(data->frame_gpu, data->frame_gpu, cv::COLOR_BGR2GRAY);
//
//    // ��������� �������� ��������
//    cv::cuda::bilateralFilter(data->frame_gpu, data->frame_gpu, 7, 10.0, 10.0);
//
//    // �������� ������������ ���� ������� �� CPU
//    data->frame_gpu.download(data->frame_cpu);
//}
//
//// ������� ��� ��������� ������ �� ������
//void drawText(FrameData* data)
//{
//    // ������ ������� � ������
//    std::unique_lock<std::mutex> lock(data->mtx);
//
//    // ������ ����� �� �����
//    cv::putText(data->frame_cpu, "Hello from GPU!", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
//}
//
//int main()
//{
//    // ��������� ������
//    cv::VideoCapture cap(0); // 0 - ������ ������
//
//    if (!cap.isOpened()) {
//        std::cerr << "Error opening camera!" << std::endl;
//        return -1;
//    }
//
//    // ������ ��������� ��� �������� ������ � �����
//    FrameData data;
//
//    while (true) {
//        // ����������� ����
//        cap >> data.frame_cpu;
//
//        // �������� ���� �� GPU
//        data.frame_gpu.upload(data.frame_cpu);
//
//        // ��������� ��������� ����� � ��������� ������
//        std::thread processingThread(processFrame, &data);
//
//        // ������ ����� �� �����
//        drawText(&data);
//
//        // ���������� ���������
//        cv::imshow("Camera Feed", data.frame_cpu);
//
//        // ������� ���������� ������ ���������
//        // 
//        processingThread.join();
//
//
//        // ������� ������� ������� 'Esc' ��� ������
//        if (cv::waitKey(1) == 27) {
//            break;
//        }
//    }
//
//    // ����������� �������
//    cap.release();
//    cv::destroyAllWindows();
//
//    return 0;
//}
