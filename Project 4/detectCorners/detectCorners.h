/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * Header file for detectCorners.cpp
 */


#ifndef DETECTCORNERS_H
#define DETECTCORNERS_H

#include <opencv2/opencv.hpp>
#include <vector>

// function declarations
int getCorners(const cv::Size chessboardSize, std::vector<cv::Point2f> &cornerSet, cv::Mat &image, bool &cornersFound);

#endif // DETECTCORNERS_H