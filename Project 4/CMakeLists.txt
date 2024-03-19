cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for getting calibration data
add_executable(calibrate getCalibration.cpp detectCorners/detectCorners.cpp calibration/calibration.cpp)
target_link_libraries(calibrate ${OpenCV_LIBS}) 

# Executable for augmented reality
add_executable(AR augmentedReality.cpp detectCorners/detectCorners.cpp calibration/calibration.cpp)
target_link_libraries(AR ${OpenCV_LIBS}) 

# Executable for augmented reality
add_executable(AR2 augmentedRealityMultiple.cpp detectCorners/detectCorners.cpp calibration/calibration.cpp)
target_link_libraries(AR2 ${OpenCV_LIBS}) 

# Executable for robust feature
add_executable(feature detectFeatures.cpp)
target_link_libraries(feature ${OpenCV_LIBS}) 

# Executable for robust feature
add_executable(generateAR generateAR.cpp detectCorners/detectCorners.cpp calibration/calibration.cpp)
target_link_libraries(generateAR ${OpenCV_LIBS}) 