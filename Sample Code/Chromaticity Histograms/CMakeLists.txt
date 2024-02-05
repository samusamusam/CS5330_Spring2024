cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for colors.cpp
add_executable(hist makeHist.cpp)
target_link_libraries(hist ${OpenCV_LIBS})