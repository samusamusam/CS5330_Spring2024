/*
  Samuel Lee
  Spring 2024
  CS 5330
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <../csv_util/csv_util.h>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**
 * This function calculates the feature vector for a given image by getting
 * the middle 7x7 pixel BGR values and storing them in a vector of float values.
 * img - the image to get the features for
 * features - the features of the vector to be stored
 */
int feature7x7(vector<Vec3b> &img, vector<float> &features)
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
      for (int k = 0; k < 3; k++)
      {
        features.push_back(img[rowStart + i][colStart + j][k]);
      }
    }
  }

  return 0; // success
}

/**
 * This function creates the feature csv files given the directory of images.
 * The following csv files are created:
 *  1. feature7x7.csv
 *  2. 
 * dirname - the name of the directory
 */
int createFeatureCSVFiles(const char *dirname)
{
  // declare feature CSV file names
  char *feature7x7CSV = "../features/feature7x7.csv";

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
    return(-1);
  }

  // initialize file index counter
  int fileIndex = 0;

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
      append_image_data_csv(feature7x7CSV, dp->d_name, feature7x7vector, fileIndex == 0 ? 1 : 0);
    }
    // add one to the file index
    fileIndex++;
  }
  if(!csvCreated) {
    return -2;
  }

  return (0);
}