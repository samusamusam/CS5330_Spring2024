/*
  Samuel Lee
  Spring 2024
  CS 5330

  Functions in this file generate top matches for a target image based on the CSV of the feature vectors.
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
 * This function gets the top matching images to the target image based on the middle 7x7 pixels
 * img - target image
 * targetImagePath - path of the target image
 * numMatches - number of matches to return
 * matches - stores the top matches found as file paths to the images
 */
int features7x7Matching(Mat &img, string &targetImagePath, int numMatches, vector<string> &matches)
{
  char *csvFilePath = "../features/feature7x7.csv"; // name of csv file
  matches.clear(); // clear matches
  // index for both imgFileNames and imgFeatureData will be the same
  vector<char *> imgFileNames; // image file names
  vector<vector<float>> imgFeatureData; // image feature data
  read_image_data_csv(csvFilePath, imgFileNames, imgFeatureData, 0); // read the csv file of the features
  priority_queue<pair<float, int>, vector<pair<float, int>>, less<pair<float, int>>> smallestSSD; // initialize priority queue

  // get feature vector data for the target image
  vector<float> targetImgFeatureData;
  feature7x7(img, targetImgFeatureData);

  // find SSD between each image
  for (int i = 0; i < imgFeatureData.size(); i++)
  {
    vector<float> singleImgFeatureData = imgFeatureData[i];
    float singleSSD = 0.0;
    for (int j = 0; j < 147; j++)
    {
      singleSSD += ((singleImgFeatureData[j] - targetImgFeatureData[j]) * (singleImgFeatureData[j] - targetImgFeatureData[j]));
    }
    // push the singleSSD to the PQ and pop the greatest value if the size of the PQ exceeds numMatches
    if (!(singleSSD == 0.0 && imgFileNames[i] == targetImagePath))
    {

      smallestSSD.push({singleSSD, i});
      if (smallestSSD.size() > numMatches)
      {
        smallestSSD.pop();
      }
    }
  }

  // get file name matches
  while (!smallestSSD.empty())
  {
    matches.push_back(imgFileNames[smallestSSD.top().second]);
    cout << "Match found: " + string(imgFileNames[smallestSSD.top().second]) << endl;
    smallestSSD.pop();
  }

  return 0;
}

/**
 * This function gets the top matching images to the target image based on the histogram feature
 * img - target image
 * targetImagePath - path of the target image
 * numMatches - number of matches to return
 * matches - stores the top matches found as file paths to the images
 */
int featuresHistMatching(Mat &img, string &targetImagePath, int numMatches, vector<string> &matches)
{
  char *csvFilePath = "../features/featureHist.csv"; // name of csv file
  matches.clear(); // clear matches
  // index for both imgFileNames and imgFeatureData will be the same
  vector<char *> imgFileNames; // image file names
  vector<vector<float>> imgFeatureData; // image feature data
  read_image_data_csv(csvFilePath, imgFileNames, imgFeatureData, 0); // read the csv file of the features
  priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> greatestIntersection; // initialize priority queue

  // get feature vector data for the target image
  vector<float> targetImgFeatureData;
  featureHist(img, targetImgFeatureData);

  // find distance between each image's histogram by finding the total intersection
  for (int i = 0; i < imgFeatureData.size(); i++)
  {
    vector<float> singleImgFeatureData = imgFeatureData[i];
    // when the two histograms are of different size
    if(imgFeatureData[i].size() != targetImgFeatureData.size()) {
      cout << "The histogram for image " + string(imgFileNames[i]) + " is a different size compared to the target histogram." << endl;
      continue;
    }

    // find intersection between two histograms
    float intersection = 0.0;
    for(int j = 0; j < targetImgFeatureData.size(); j++) {
      intersection += min(targetImgFeatureData[j],singleImgFeatureData[j]);
    }

    // push the intersection to the PQ and pop the greatest value if the size of the PQ exceeds numMatches
    if (!(imgFileNames[i] == targetImagePath))
    {
      greatestIntersection.push({intersection, i});
      if (greatestIntersection.size() > numMatches)
      {
        greatestIntersection.pop();
      }
    }
  }

  // get file name matches
  while (!greatestIntersection.empty())
  {
    matches.push_back(imgFileNames[greatestIntersection.top().second]);
    cout << "Match found: " + string(imgFileNames[greatestIntersection.top().second]) << endl;
    greatestIntersection.pop();
  }

  return 0;
}

/**
 * This function gets the top matching images to the target image based on the multi histogram feature
 * img - target image
 * targetImagePath - path of the target image
 * numMatches - number of matches to return
 * matches - stores the top matches found as file paths to the images
 */
int featuresMultiHistMatching(Mat &img, string &targetImagePath, int numMatches, vector<string> &matches)
{
  char *csvFilePath = "../features/featureMultiHist.csv"; // name of csv file
  matches.clear(); // clear matches
  // index for both imgFileNames and imgFeatureData will be the same
  vector<char *> imgFileNames; // image file names
  vector<vector<float>> imgFeatureData; // image feature data
  read_image_data_csv(csvFilePath, imgFileNames, imgFeatureData, 0); // read the csv file of the features
  priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> greatestIntersection; // initialize priority queue

  // get feature vector data for the target image
  vector<float> targetImgFeatureData;
  featureMultiHist(img, targetImgFeatureData);

  // find distance between each image's histogram by finding the total intersection
  for (int i = 0; i < imgFeatureData.size(); i++)
  {
    vector<float> singleImgFeatureData = imgFeatureData[i];
    // when the two histograms are of different size
    if(imgFeatureData[i].size() != targetImgFeatureData.size()) {
      cout << "The histogram for image " + string(imgFileNames[i]) + " is a different size compared to the target histogram." << endl;
      continue;
    }

    // find intersection between two histograms
    float intersection = 0.0;
    for(int j = 0; j < targetImgFeatureData.size(); j++) {
      intersection += min(targetImgFeatureData[j],singleImgFeatureData[j]);
    }

    // push the intersection to the PQ and pop the greatest value if the size of the PQ exceeds numMatches
    if (!(imgFileNames[i] == targetImagePath))
    {
      greatestIntersection.push({intersection, i});
      if (greatestIntersection.size() > numMatches)
      {
        greatestIntersection.pop();
      }
    }
  }

  // get file name matches
  while (!greatestIntersection.empty())
  {
    matches.push_back(imgFileNames[greatestIntersection.top().second]);
    cout << "Match found: " + string(imgFileNames[greatestIntersection.top().second]) << endl;
    greatestIntersection.pop();
  }

  return 0;
}

/**
 * This function gets the top matching images to the target image based on color and texture
 * img - target image
 * targetImagePath - path of the target image
 * numMatches - number of matches to return
 * matches - stores the top matches found as file paths to the images
 */
int featuresColorTextureMatching(Mat &img, string &targetImagePath, int numMatches, vector<string> &matches)
{
  char *csvFilePath = "../features/featureColorTexture.csv"; // name of csv file
  matches.clear(); // clear matches
  // index for both imgFileNames and imgFeatureData will be the same
  vector<char *> imgFileNames; // image file names
  vector<vector<float>> imgFeatureData; // image feature data
  read_image_data_csv(csvFilePath, imgFileNames, imgFeatureData, 0); // read the csv file of the features
  priority_queue<pair<float, int>, vector<pair<float, int>>, less<pair<float, int>>> smallestSSD; // initialize priority queue

  // get feature vector data for the target image
  vector<float> targetImgFeatureData;
  featureColorTextureHist(img, targetImgFeatureData);

  // find SSD between each image
  for (int i = 0; i < imgFeatureData.size(); i++)
  {
    vector<float> singleImgFeatureData = imgFeatureData[i];
    float singleSSD = 0.0;
    for (int j = 0; j < 147; j++)
    {
      singleSSD += ((singleImgFeatureData[j] - targetImgFeatureData[j]) * (singleImgFeatureData[j] - targetImgFeatureData[j]));
    }
    // push the singleSSD to the PQ and pop the greatest value if the size of the PQ exceeds numMatches
    if (!(singleSSD == 0.0 && imgFileNames[i] == targetImagePath))
    {

      smallestSSD.push({singleSSD, i});
      if (smallestSSD.size() > numMatches)
      {
        smallestSSD.pop();
      }
    }
  }

  // get file name matches
  while (!smallestSSD.empty())
  {
    matches.push_back(imgFileNames[smallestSSD.top().second]);
    cout << "Match found: " + string(imgFileNames[smallestSSD.top().second]) << endl;
    smallestSSD.pop();
  }

  return 0;
}

/**
 * This function gets the top matching images to the target image based on the ResNet18 feature
 * img - target image
 * targetImageName - name of the target image
 * numMatches - number of matches to return
 * matches - stores the top matches found as file paths to the images
 */
int featuresDenMatching(Mat &img, string &targetImageName, int numMatches, vector<string> &matches)
{
  char *csvFilePath = "../features/ResNet18_olym.csv"; // name of csv file
  matches.clear(); // clear matches
  // index for both imgFileNames and imgFeatureData will be the same
  vector<char *> imgFileNames; // image file names
  vector<vector<float>> imgFeatureData; // image feature data
  read_image_data_csv(csvFilePath, imgFileNames, imgFeatureData, 0); // read the csv file of the features
  priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> smallestSSD; // initialize priority queue

  // get feature vector data for the target image
  vector<float> targetImgFeatureData;
  for (int i = 0; i < imgFeatureData.size(); i++) {
    if(targetImageName == imgFileNames[i]) {
      targetImgFeatureData = imgFeatureData[i];
    }
  }

  // find SSD between each image
  for (int i = 0; i < imgFeatureData.size(); i++)
  {
    vector<float> singleImgFeatureData = imgFeatureData[i];
    float singleSSD = 0.0;
    for (int j = 0; j < targetImgFeatureData.size(); j++)
    {
      singleSSD += ((singleImgFeatureData[j] - targetImgFeatureData[j]) * (singleImgFeatureData[j] - targetImgFeatureData[j]));
    }
    // push the singleSSD to the PQ and pop the greatest value if the size of the PQ exceeds numMatches
    if (!(imgFileNames[i] == targetImageName))
    {

      smallestSSD.push({singleSSD, i});
      if (smallestSSD.size() > numMatches)
      {
        smallestSSD.pop();
      }
    }
  }

  // get file name matches
  while (!smallestSSD.empty())
  {
    matches.push_back(imgFileNames[smallestSSD.top().second]);
    cout << "Match found: " + string(imgFileNames[smallestSSD.top().second]) << endl;
    smallestSSD.pop();
  }

  return 0;
}

