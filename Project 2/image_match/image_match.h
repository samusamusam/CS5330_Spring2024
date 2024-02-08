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

int features7x7Matching(const Mat &img, string &targetImagePath, int numMatches, vector<string> &matches);
int featuresHistMatching(const Mat &img, string &targetImagePath, int numMatches, vector<string> &matches);
#endif