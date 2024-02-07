/*
  Samuel Lee
  Spring 2024
  CS 5330

 */

#ifndef COMPUTE_FEATURE_H
#define COMPUTE_FEATURE_H

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int feature7x7(vector<Vec3b> &img, vector<float> &features);

int createFeatureCSVFiles(const char *dirname);

#endif