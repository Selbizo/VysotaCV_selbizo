#pragma once

#include <string>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
// #include <filesystem>

void measuring(cuda::GpuMat oldFrame, cuda::GpuMat Frame, Ptr<cuda::CornersDetector> detector, double tauStab, double kSwitch, Rect roi, string name_dir);
void writeRes(Mat TStab, Point2f d, string dirName);
