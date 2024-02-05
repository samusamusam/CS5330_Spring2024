cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for colors.cpp
add_executable(kmeans colors.cpp kmeans.cpp)
target_link_libraries(kmeans ${OpenCV_LIBS})