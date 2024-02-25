cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for Object Recognition program
add_executable(main main.cpp threshold.cpp cleanup.cpp segment.cpp computeFeatures.cpp csvReadWrite.cpp distance.cpp)
target_link_libraries(main ${OpenCV_LIBS})