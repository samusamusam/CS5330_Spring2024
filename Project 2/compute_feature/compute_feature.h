/*
  Samuel Lee
  Spring 2024
  CS 5330

  Header file for compute_feature.cpp
 */

#ifndef COMPUTE_FEATURE_H
#define COMPUTE_FEATURE_H

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int feature7x7(Mat &img, vector<float> &features);
int featureHist(Mat &img, vector<float> &features);
int featureMultiHist(Mat &img, vector<float> &features);
int featureColorTextureHist(Mat &img, vector<float> &features);
int createFeatureCSVFiles(char *dirname);

#endif