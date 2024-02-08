/*
  Samuel Lee
  Spring 2024
  CS 5330

  Functions in this file calculate the features based on a specific method and store the features in a CSV file.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include "../csv_util/csv_util.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**
 * This function calculates the feature vector for a given image by getting
 * the middle 7x7 pixel BGR values and storing them in a vector of float values.
 * img - the image to get the features for
 * features - the features of the vector to be stored
 */
int feature7x7(const Mat &img, vector<float> &features)
{
  features.clear(); // empty features vector

  // if there isn't enough spacee to extract a 7x7 feature vector
  if (img.rows < 7 || img.cols < 7)
  {
    cerr << "Error: file size cannot be less than 7x7 pixels." << endl;
    return -1;
  }

  // find start points as reference for the 7x7 pixel
  int rowStart = (img.rows / 2) - 3;
  int colStart = (img.cols / 2) - 3;

  // loop through each of the 7x7 center pixels and add the BGR features
  for (int i = 0; i < 7; i++)
  {
    for (int j = 0; j < 7; j++)
    {
      Vec3b pixel = img.at<Vec3b>(rowStart + i, colStart + j);
      for (int k = 0; k < 3; k++)
      {
        features.push_back(pixel[k]);
      }
    }
  }
  return 0; // success
}

/**
 * This function calculates 2D histogram features and stores them in a vector of floats
 * img - the image to get the features for
 * features - the features of the vector to be stored
 */
int histFeature(const Mat &img, vector<float> &features)
{
  features.clear(); // empty features vector
  
  int histsize = 16; // bin size

  hist = Mat::zeros( Size( histsize, histsize ), CV_32FC1 ); // 2-D histogram 

  // loop through each pixel
  for(int i = 0; i < img.rows; i++) {
    Vec3b *ptr = img.ptr<Vec3b>(i); // point to row i
    for(int j = 0; j < img.cols; j++) {
      // get BGR values
      float B = ptr[j][0]; // blue
      float G = ptr[j][1]; // green
      float R = ptr[j][2]; // red

      // compute r,g chromaticity
      float divisor = B+G+R;
      divisor = divisor > 0 ? divisor : 1.0; // if divisor = 0, assign 1.0 as default

      // normalize r and g based on divisor for weights of each relative to all channels
      float r = R / divisor;
      float g = G / divisor;

      // compute index
      int rindex = (int)(r * (histsize-1) + 0.5);
      int gindex = (int)(g * (histsize-1) + 0.5);

      // increment histogram at respective r and g index
      hist.at<float>(rindex,gindex)++;
    }
  }
  
  hist /= (img.rows * img.cols); // normalizes all values of the histogram

  // loop through each value of the histogram and store the value in a vector
  for(int i = 0; i < hist.rows; i++) {
    float* ptr = hist.ptr<float>(i);
    for(int j = 0; j < hist.cols; j++) {
      features.push_back(ptr[j]);
    }
  }

  return 0; // success
}


/**
 * This function creates the feature csv files given the directory of images.
 * The following csv files are created:
 *  1. feature7x7.csv
 *  2. featureHist.csv
 *  3. 
 * dirname - the name of the directory
 */
int createFeatureCSVFiles(char *dirname)
{
  // declare feature CSV file names
  char feature7x7CSV[] = "../features/feature7x7.csv";
  char featureHistCSV[] = "../features/featureHist.csv";

  // delete the csv files if they exist
  if (remove(feature7x7CSV) != 0 || remove(featureHistCSV))
  {
    std::cerr << "Error: Failed to delete file: " << feature7x7CSV << endl;
  }

  cout << "File " << feature7x7CSV << " has been deleted." << endl;
  cout << "File " << featureHistCSV << " has been deleted." << endl;

  // declare variables for reading the image files
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // get the directory path
  printf("Processing directory %s\n", dirname);

  // open the directory
  dirp = opendir(dirname);
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    return (-1);
  }

  // initialize boolean flag for whether the csv files were created
  bool csvCreated = false;

  // loop over all the files in the image file listing
  while ((dp = readdir(dirp)) != NULL)
  {

    // check if the file is an image
    if (strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif"))
    {
      csvCreated = true;
      printf("processing image file: %s\n", dp->d_name);

      // vectors for features
      vector<float> feature7x7vector;
      vector<float> histFeatureVector;

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      // print full file path
      printf("full path name: %s\n", buffer);

      // read current directory image
      Mat currentImg = imread(buffer);

      // calculate the 7x7 feature and append it to the csv file
      feature7x7(currentImg, feature7x7vector);
      append_image_data_csv(feature7x7CSV, buffer, feature7x7vector, 0);

      // calculate the histogram feature and append it to the csv file
      histFeature(currentImg, histFeatureVector);
      append_image_data_csv(featureHistCSV, buffer, histFeatureVector, 0);
    }
  }
  if (!csvCreated)
  {
    return -2;
  }

  return 0;
}