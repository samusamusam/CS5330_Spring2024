/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Include file for a library of filters
*/

#ifndef GAUSS_H
#define GAUSS_H

// Implements a 3x3 Gaussian filter using the at<> method
int gauss3x3at( cv::Mat &src, cv::Mat &dst );

// Implements a 3x3 Gaussian filter using the ptr<> method
int gauss3x3ptr( cv::Mat &src, cv::Mat &dst );

#endif
