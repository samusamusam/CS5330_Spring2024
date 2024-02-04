/**
 * Samuel Lee
 * Spring 2024
 * CS 5330
 * 
 * Include file for a library of filters
*/

#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "opencv2/opencv.hpp"

// Implement file for a library of filters
int gauss3x3at( cv::Mat &src, cv::Mat &dst );

int gauss3x3ptr( cv::Mat &src, cv::Mat &dst );
#endif