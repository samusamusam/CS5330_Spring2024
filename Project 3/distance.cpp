/*
	Samuel Lee
	Anjith Prakash Chathan Kandy
	2/23/24
	This file contains functions to get distance and closest labels
*/

#include <cstdio>
#include <cstring>
#include <cfloat>
#include <iostream>
#include "FeatureData.h"

using namespace std;

/**
 * This function calculates the mean of the doubles in the vector
 * data - data to get mean of
*/
double calculateMean(vector<double> &data) {
  double sum = 0.0;
  for (double value : data) {
    sum+=value;
  }
  return sum / data.size();
}

/**
 * This function calcualtes the standard deviation of the given data set
 * data - data to get the standard deviation of
*/
double calculate_stdev(vector<double> &data) {
  double mean = calculateMean(data);
  double variance = 0.0;
  for (double value : data) {
    variance += pow(value-mean, 2);
  }

  variance /= data.size();
  return sqrt(variance);
}

/**
 * Distance formula between two vectors via euclidean distance
 * curr - feature vector #1
 * other - feature vector #2
 * stdev - vector of the standard deviations of the features
 */
double distanceScaledEuclidean(FeatureData &curr, FeatureData &other, vector<double> stdev)
{
  if(stdev.size() < 2) {
    cerr << "More standard deviations need to be calculated first." << endl;
    return -1.0;
  }
  for(int i = 0; i < stdev.size(); i++) {
    if(stdev.at(i) == 0.0) {
      stdev.at(i) = 1.0;
    }
  }
  return sqrt(pow((curr.f1-other.f1)/stdev.at(0),2) + pow((curr.f2-other.f2)/stdev.at(1),2));
}

/**
 * Distance formula between two vectors via cosine similarity
 * curr - feature vector #1
 * other - feature vector #2
 */
double cosineSimilarity(const FeatureData &curr, const FeatureData &other) {
    double dotProduct = (curr.f1 * other.f1) + (curr.f2 * other.f2);

    double magnitudeCurr = sqrt(pow(curr.f1, 2) + pow(curr.f2, 2));
    double magnitudeOther = sqrt(pow(other.f1, 2) + pow(other.f2, 2));

    if (magnitudeCurr == 0 || magnitudeOther == 0) {
        return 0.0;
    }

    return dotProduct / (magnitudeCurr * magnitudeOther);
}


/**
 * This function returns the label of the closest feature vector to the current feature vector.
 * curr - current feature vector
 * trainingData - database of feature vectors and corresponding labels
 * threshold - threshold to check if object is identified or not
 * stdev - vector of the standard deviations of the features
 */
string closestLabelScaledEuclidean(FeatureData &curr, vector<FeatureData> &trainingData, double threshold, vector<double> stdev)
{

  FeatureData closest;
  double closestDist = DBL_MAX;
  double currDist;

  // find and update closest FeatureData, closestDist
  for (int i = 0; i < trainingData.size(); i++)
  {
    currDist = distanceScaledEuclidean(curr, trainingData.at(i), stdev);
    if (currDist < closestDist)
    {
      closestDist = currDist;
      closest = trainingData.at(i);
    }
  }

  // if closest distance is above threshold, unknown object
  if(closestDist > threshold) {
    return "UNKNOWN OBJECT";
  }

  return closest.label;
}

/**
 * This function returns the label of the closest feature vector to the current feature vector.
 * curr - current feature vector
 * trainingData - database of feature vectors and corresponding labels
 * threshold - threshold to check if object is identified or not
 */
string closestLabelCosineSimilarity(FeatureData &curr, vector<FeatureData> &trainingData, double threshold)
{

  FeatureData mostSimilar;
  double largestSimilarity = DBL_MIN;
  double currSimilarity;

  // find and update closest FeatureData, closestDist
  for (int i = 0; i < trainingData.size(); i++)
  {
    currSimilarity = cosineSimilarity(curr, trainingData.at(i));
    if (currSimilarity > largestSimilarity)
    {
      largestSimilarity = currSimilarity;
      mostSimilar = trainingData.at(i);
    }
  }

  // if closest distance is above threshold, unknown object
  if(largestSimilarity < threshold) {
    return "UNKNOWN OBJECT";
  }

  return mostSimilar.label;
}

/**
 * This calculates the labels of the nearest k labels via KNN
 * curr - current feature data
 * trainingData - database of feature data
 * stdev - standard deviation of 
 * k - how many values to get
*/
vector<string> kNearestLabels(FeatureData& curr, vector<FeatureData>& trainingData, vector<double> stdev, int k) {
    vector<pair<double, string>> distances;

    // calculate distances
    for (FeatureData& data : trainingData) {
        double distance = distanceScaledEuclidean(curr, data, stdev);
        distances.push_back({ distance, data.label });
    }

    // sort distances
    sort(distances.begin(), distances.end());

    // extract labels of the k nearest neighbors without voting
    vector<string> nearestLabels;
    for (int i = 0; i < k && i < distances.size(); i++) {
        nearestLabels.push_back(distances[i].second);
    }

    return nearestLabels;
}