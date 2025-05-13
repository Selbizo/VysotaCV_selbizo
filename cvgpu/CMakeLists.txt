cmake_minimum_required(VERSION 3.10)
project(cuda_opencv_stab)

find_package(PkgConfig REQUIRED)
find_package (CUDA REQUIRED)

set(ENV{PKG_CONFIG_PATH} ${PKGPATH})
set(ENV{OpenCV_DIR} ${OpenCV_DIR})

pkg_check_modules(OpenCV REQUIRED opencv4)

set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}  -std=c++17 ")
set(OpenCV_LIBRARIES, /media/jetson/EMMCdrive/cvbuild/lib/)

include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui opencv_cudaarithm opencv_cudaoptflow)
add_executable(cuda_opencv_stab, main.cpp Config.hpp measuring.cpp measuring.hpp stabilizationFunctions.hpp wienerFilter.hpp)

target_link_libraries(cuda_opencv_stab ${OpenCV_LIBRARIES}  -pthread)


message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
