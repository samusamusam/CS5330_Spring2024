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
int featureColorTextureDNNHist(Mat &img, vector<float> &features, string imgName, vector<char *> &imgFileNames, vector<vector<float>> &imgFeatureData);
int createFeatureCSVFiles(char *dirname, char *feature7x7CSV, char *featureHistCSV,
                          char *featureMultiHistCSV, char *featureColorTextureHistCSV,
                          char *featureColorTextureDNNHistCSV, char *featureFaceCSV,
                          vector<char *> &imgFileNames, vector<vector<float>> &imgFeatureData);
int featureFaces(Mat &img, char *imgFileName, char *featureCSV);
int featureFirstFace(Mat &img, vector<float> &features);
                    
#endif