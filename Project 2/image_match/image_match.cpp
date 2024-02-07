/*
  Samuel Lee
  Spring 2024
  CS 5330
*/

#include <queue>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include "../csv_util/csv_util.h"
#include "../compute_feature/compute_feature.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**
 * This function gets the top matching images to the target image 
 * based on the features pre-calculated in the CSV file.
 * img - target image
 * csvFilePath - csv file path
 * numMatches - number of matches to return
 * matches - stores the top matches found as file paths to the images
*/
int features7x7Matching(const vector<Vec3b> &img, string &targetImagePath, int numMatches, vector<string> &matches) {
  // read csv file and store values 
  string csvFilePath = "../features/feature7x7.csv";
  matches.clear();
  vector<char *> imgFileNames;
  vector<vector<float>> imgFeatureData;
  map<int, float> sortedSSD;
  read_image_data_csv(csvFilePath, imgFileNames, imgFeatureData, 0);
  priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> smallestSSD;

  // get feature vector data for the target image
  vector<float> targetImgFeatureData;
  feature7x7(img, targetImgFeatureData);

  // find SSD between each image
  for(int i = 0; i < imgFeatureData.size(); i++) {
    vector<float> singleImgFeatureData = imgFeatureData[i];
    float singleSSD = 0.0;
    for(int j = 0; j < 147; j++) {
      singleSSD += ((singleImgFeatureData[j] - targetImgFeatureData[j]) * (singleImgFeatureData[j] - targetImgFeatureData[j]));
    }
    // push the singleSSD to the PQ and pop the greatest value if the size of the PQ exceeds numMatches
    smallestSSD.push({singleSSD, i});
    if (smallestSSD.size() > numMatches) {
      smallestSSD.pop();
    }
  }
  
  // get file name matches
  while (!smallestSSD.empty()) {
    matches.push_back(imgFileNames[smallestSSD.top().second]);
    smallestSSD.pop();
  }

  return 0;
}