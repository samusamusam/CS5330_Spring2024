/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/22/24
	Header file to support segmentation
*/

#ifndef SEGMENT_H
#define SEGMENT_H

#include <opencv2/opencv.hpp>

// function declarations
cv::Mat labelConnectedComponents(cv::Mat &src, int minSize, cv::Mat &stats, int &num_labels, int maxRegions, std::priority_queue<std::pair<double, int>> &minDistances);
cv::Mat colorlabelConnectedComponents(cv::Mat &labels, const std::vector<cv::Vec3b> &colors, std::priority_queue<std::pair<double, int>> &minDistances);

#endif // SEGMENT_H