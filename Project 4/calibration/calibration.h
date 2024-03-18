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
int getChessboardWorldPoints(std::vector<cv::Point3f> &point_set, const cv::Size &chessboardSize);
int project3DAxes(cv::Mat &rotations, cv::Mat &translations, cv::Mat &cameraMat, cv::Mat &distortionCoeffs, cv::Mat &img);
int projectShape3D(cv::Mat &rotations, cv::Mat &translations, cv::Mat &cameraMat, cv::Mat &distortionCoeffs, cv::Mat &img);

#endif // CALIBRATION_H