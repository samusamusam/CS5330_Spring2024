/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * Header file for calibration.cpp
 */

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <vector>

// function declarations
int getChessboardWorldPointsVec3f(std::vector<cv::Vec3f> &point_set, const cv::Size &chessboardSize);
int getChessboardWorldPointsPoint3f(std::vector<cv::Point3f> &point_set, const cv::Size &chessboardSize);

#endif // CALIBRATION_H