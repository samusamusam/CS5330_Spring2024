cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for tutorial 1
add_executable(Tutorial1 Examples/Tutorial1.cpp)
target_link_libraries(Tutorial1 ${OpenCV_LIBS})

# Executable for tutorial 2
add_executable(Tutorial2 Examples/Tutorial2.cpp Examples/Gaussian.cpp)
target_link_libraries(Tutorial2 ${OpenCV_LIBS})

# Executable for imgDisplay.cpp
add_executable(imageDisplay imgDisplay.cpp)
target_link_libraries(imageDisplay ${OpenCV_LIBS})

# Executable for vidDisplay.cpp
add_executable(videoDisplay vidDisplay.cpp filter.cpp faceDetect/faceDetect.cpp)
target_link_libraries(videoDisplay ${OpenCV_LIBS})

# Executable for timeBlur.cpp
add_executable(timeBlur timeBlur.cpp filter.cpp)
target_link_libraries(timeBlur ${OpenCV_LIBS})