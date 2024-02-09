/*
  Samuel Lee
  Spring 2024
  CS 5330

  Header file for image_match.cpp
 */

#ifndef IMAGE_MATCH_H
#define IMAGE_MATCH_H

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int features_match_SSD(Mat &img, string &targetImageName, int numMatches, vector<string> &matches, char *csvFilePath,
                       vector<float> &targetImgFeatureData, bool predefined);
int features_match_intersection(Mat &img, string &targetImageName, int numMatches, vector<string> &matches, char *csvFilePath,
                                vector<float> &targetImgFeatureData, bool predefined);
#endif