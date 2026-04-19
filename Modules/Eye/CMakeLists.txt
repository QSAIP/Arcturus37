cmake_minimum_required(VERSION 3.5)

project(run VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
set(OpenCV_DIR "/usr/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    Eye.cpp
	Eye.hpp
    RKLogger.hpp
)

add_executable(run ${PROJECT_SOURCES})
target_link_libraries(run ${OpenCV_LIBS})
