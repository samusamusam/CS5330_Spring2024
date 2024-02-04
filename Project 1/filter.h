/**
 * Samuel Lee
 * Spring 2024
 * CS 5330
 * Header include file for filter.cpp
*/

#ifndef FILTER_H
#define FILTER_H

#include "opencv2/opencv.hpp"

// custom greyscale filter
int customGreyscale( cv::Mat &src, cv::Mat &dst );

// sepia filter
int sepiaFilter( cv::Mat &src, cv::Mat &dst );

// 5 x 5 blur filter #1
int blur5x5_1( cv::Mat &src, cv::Mat &dst );

// 5 x 5 blur filter #2
int blur5x5_2( cv::Mat &src, cv::Mat &dst );

// horizontal sobel filter
int sobelX3x3( cv::Mat &src, cv::Mat &dst );

// vertical sobel filter
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

// gradient sobel filter
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

// blur quantize filter
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

// green image filter
int greenify( cv::Mat &src, cv::Mat &dst);

// median filter
int medianFilter( cv::Mat &src, cv::Mat &dst);

// face cover filter
int faceCoverFilter( cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);
#endif