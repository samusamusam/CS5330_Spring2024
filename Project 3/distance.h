/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/23/24
	Header file to support distance functions
*/

#ifndef DISTANCE_H
#define DISTANCE_H

#include "FeatureData.h"
#include <string>

// function declarations
double calculate_stdev(std::vector<double> &data);
double distanceScaledEuclidean(FeatureData &curr, FeatureData &other, std::vector<double> stdev);
double cosineSimilarity(const FeatureData &curr, const FeatureData &other);
std::string closestLabelScaledEuclidean(FeatureData &curr, std::vector<FeatureData> &trainingData, double threshold, std::vector<double> stdev);
std::string closestLabelCosineSimilarity(FeatureData &curr, std::vector<FeatureData> &trainingData, double threshold);
std::vector<std::string> kNearestLabels(FeatureData& curr, std::vector<FeatureData>& trainingData, std::vector<double> stdev, int k);

#endif // DISTANCE_H