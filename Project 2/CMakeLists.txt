cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for tutorial 1
# add_executable(Tutorial1 Examples/Tutorial1.cpp)
# target_link_libraries(Tutorial1 ${OpenCV_LIBS})