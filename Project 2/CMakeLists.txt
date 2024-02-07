cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Executable for Image Matcher program
add_executable(main main.cpp image_match/image_match.cpp compute_feature/compute_feature.cpp csv_util/csv_util.cpp)
target_link_libraries(main ${OpenCV_LIBS})