//#include <opencv2/opencv.hpp>
//#include <opencv2/cudawarping.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <iostream>
//#include <thread>
//#include <mutex>
//
//// Глобальная структура для хранения кадра и мьютекса
//struct FrameData {
//    cv::Mat frame_cpu;
//    cv::cuda::GpuMat frame_gpu;
//    std::mutex mtx;
//};
//
//// Функция для обработки кадров на GPU
//void processFrame(FrameData* data)
//{
//    // Защита доступа к кадрам
//    std::unique_lock<std::mutex> lock(data->mtx);
//
//    // Конвертируем кадр в оттенки серого
//    cv::cuda::cvtColor(data->frame_gpu, data->frame_gpu, cv::COLOR_BGR2GRAY);
//
//    // Применяем Гауссово размытие
//    cv::cuda::bilateralFilter(data->frame_gpu, data->frame_gpu, 7, 10.0, 10.0);
//
//    // Копируем обработанный кадр обратно на CPU
//    data->frame_gpu.download(data->frame_cpu);
//}
//
//// Функция для рисования текста на кадрах
//void drawText(FrameData* data)
//{
//    // Защита доступа к кадрам
//    std::unique_lock<std::mutex> lock(data->mtx);
//
//    // Рисуем текст на кадре
//    cv::putText(data->frame_cpu, "Hello from GPU!", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
//}
//
//int main()
//{
//    // Открываем камеру
//    cv::VideoCapture cap(0); // 0 - индекс камеры
//
//    if (!cap.isOpened()) {
//        std::cerr << "Error opening camera!" << std::endl;
//        return -1;
//    }
//
//    // Создаём структуру для хранения данных о кадре
//    FrameData data;
//
//    while (true) {
//        // Захватываем кадр
//        cap >> data.frame_cpu;
//
//        // Копируем кадр на GPU
//        data.frame_gpu.upload(data.frame_cpu);
//
//        // Запускаем обработку кадра в отдельном потоке
//        std::thread processingThread(processFrame, &data);
//
//        // Рисуем текст на кадре
//        drawText(&data);
//
//        // Показываем результат
//        cv::imshow("Camera Feed", data.frame_cpu);
//
//        // Ожидаем завершения потока обработки
//        // 
//        processingThread.join();
//
//
//        // Ожидаем нажатия клавиши 'Esc' для выхода
//        if (cv::waitKey(1) == 27) {
//            break;
//        }
//    }
//
//    // Освобождаем ресурсы
//    cap.release();
//    cv::destroyAllWindows();
//
//    return 0;
//}
