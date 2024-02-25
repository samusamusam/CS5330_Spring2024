/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/22/24
	Header file to support feature computing functions
*/

#ifndef COMPUTEFEATURES_H
#define COMPUTEFEATURES_H

#include <opencv2/opencv.hpp>

// function declarations
int computeFeatures(const cv::Mat &regionMap, int regionID, double &percentFilled, double &heightWidthRatio, cv::Mat &src, cv::Mat &original);

#endif // COMPUTEFEATURES_H