/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/21/24
	Header file to support cleanup of image using morphological filtering
*/


#ifndef CLEANUP_H
#define CLEANUP_H

#include <opencv2/opencv.hpp>

// function declarations
void cleanup(cv::Mat &thresholdedImage);

#endif // CLEANUP_H