cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for AR Program
add_executable(main main.cpp detectCorners/detectCorners.cpp calibration/calibration.cpp)
target_link_libraries(main ${OpenCV_LIBS}) 