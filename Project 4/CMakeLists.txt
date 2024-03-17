cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for getting calibration data
add_executable(calibrate getCalibration.cpp detectCorners/detectCorners.cpp calibration/calibration.cpp)
target_link_libraries(calibrate ${OpenCV_LIBS}) 

# Executable for augmented reality
add_executable(detect augmentedReality.cpp detectCorners/detectCorners.cpp calibration/calibration.cpp)
target_link_libraries(detect ${OpenCV_LIBS}) 