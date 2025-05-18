//конфигурация для запуска main.cpp
#pragma once
#include <fstream>
#include <iostream>

#include <opencv2/cudaoptflow.hpp> 
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



//string videoSource = "http://192.168.0.102:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://10.108.144.71:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://10.139.27.71:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://192.168.0.103:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://192.168.0.100:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/treeshd.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/treesfhd.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/trees4k.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/Pixel_Vysota_VID_1.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/Pixel_Vysota_park4k.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/Pixel_Vysota_parkfhd60fps.mp4"; // pad6-100, pixel4-101, pixel-102
string videoSource = "./SourceVideos/testVelocityEstimator.mp4"; // pad6-100, pixel4-101, pixel-102
//int videoSource = 0;

bool writeVideo = false;
bool stabPossible = false;

const int compression = 2; //коэффициент сжатия изображения для обработки //4k 1->26ms 2->20ms 3->20ms

//переменные для поиска характерных точек
int	srcType = CV_8UC1;
int maxCorners = 400 / compression; //100/n
double qualityLevel = 0.003 / compression; //0.0001
double minDistance = 6.0 / compression + 2.0; //8.0
int blockSize = 40 / compression + 8; //45 80 максимальное значение окна
bool useHarrisDetector = true;
double harrisK = qualityLevel;

// переменные для оптического потока
bool useGray = true;
int winSize = blockSize;
int maxLevel = 3 + 8/compression;
int iters = 10;
double minDist = minDistance;