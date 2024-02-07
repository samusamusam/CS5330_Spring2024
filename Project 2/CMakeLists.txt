cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for Image Matcher GUI
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})