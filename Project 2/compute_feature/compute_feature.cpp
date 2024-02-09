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
int feature7x7(Mat &img, vector<float> &features)
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
int featureHist(Mat &img, vector<float> &features)
{
  features.clear();  // empty features vector
  Mat hist;          // initialize histogram
  int histsize = 16; // bin size

  hist = Mat::zeros(Size(histsize, histsize), CV_32FC1); // 2-D histogram

  // loop through each pixel
  for (int i = 0; i < img.rows; i++)
  {
    Vec3b *ptr = img.ptr<Vec3b>(i); // point to row i
    for (int j = 0; j < img.cols; j++)
    {
      // get BGR values
      float B = ptr[j][0]; // blue
      float G = ptr[j][1]; // green
      float R = ptr[j][2]; // red

      // compute r,g chromaticity
      float divisor = B + G + R;
      divisor = divisor > 0 ? divisor : 1.0; // if divisor = 0, assign 1.0 as default

      // normalize r and g based on divisor for weights of each relative to all channels
      float r = R / divisor;
      float g = G / divisor;

      // compute index
      int rindex = (int)(r * (histsize - 1) + 0.5);
      int gindex = (int)(g * (histsize - 1) + 0.5);

      // increment histogram at respective r and g index
      hist.at<float>(rindex, gindex)++;
    }
  }

  hist /= (img.rows * img.cols); // normalizes all values of the histogram

  // loop through each value of the histogram and store the value in a vector
  for (int i = 0; i < hist.rows; i++)
  {
    float *ptr = hist.ptr<float>(i);
    for (int j = 0; j < hist.cols; j++)
    {
      features.push_back(ptr[j]);
    }
  }

  return 0; // success
}

/**
 * This function calculates 2D histogram features based on top and
 * bottom halves and stores them in a vector of floats
 * img - the image to get the features for
 * features - the features of the vector to be stored
 */
int featureMultiHist(Mat &img, vector<float> &features)
{
  features.clear();  // empty features vector
  int histsize = 16; // bin size

  Mat topHalfHist = Mat::zeros(Size(histsize, histsize), CV_32FC1);    // top half histogram
  Mat bottomHalfHist = Mat::zeros(Size(histsize, histsize), CV_32FC1); // bottom half histogram

  // loop through each pixel
  for (int i = 0; i < img.rows; i++)
  {
    Vec3b *ptr = img.ptr<Vec3b>(i); // point to row i
    for (int j = 0; j < img.cols; j++)
    {
      // get BGR values
      float B = ptr[j][0]; // blue
      float G = ptr[j][1]; // green
      float R = ptr[j][2]; // red

      // compute r,g chromaticity
      float divisor = B + G + R;
      divisor = divisor > 0 ? divisor : 1.0; // if divisor = 0, assign 1.0 as default

      // normalize r and g based on divisor for weights of each relative to all channels
      float r = R / divisor;
      float g = G / divisor;

      // compute index
      int rindex = (int)(r * (histsize - 1) + 0.5);
      int gindex = (int)(g * (histsize - 1) + 0.5);

      // increment histogram at respective r and g index based on top/bottom half
      if (i < img.rows / 2)
      {
        topHalfHist.at<float>(rindex, gindex)++; // top half
      }
      else
      {
        bottomHalfHist.at<float>(rindex, gindex)++; // bottom half
      }
    }
  }

  topHalfHist /= (img.rows / 2 * img.cols);    // normalizes all values of the top half histogram
  bottomHalfHist /= (img.rows / 2 * img.cols); // normalizes all values of the bottom half histogram

  // loop through each value of the histogram and store the value in a vector
  for (int i = 0; i < topHalfHist.rows; i++)
  {
    float *ptr = topHalfHist.ptr<float>(i);
    for (int j = 0; j < topHalfHist.cols; j++)
    {
      features.push_back(ptr[j]);
    }
  }

  for (int i = 0; i < bottomHalfHist.rows; i++)
  {
    float *ptr = bottomHalfHist.ptr<float>(i);
    for (int j = 0; j < bottomHalfHist.cols; j++)
    {
      features.push_back(ptr[j]);
    }
  }
  return 0; // success
}

/**
 * This function calculates the feature vector via a color histogram and a texture histogram
 * img - the image to get the features for
 * features - the features of the vector to be stored
 */
int featureColorTextureHist(Mat &img, vector<float> &features)
{
  features.clear();                            // empty features vector
  int histsize = 16;                           // bin size
  Mat greyscaleImg;                            // greyscale image
  cvtColor(img, greyscaleImg, COLOR_BGR2GRAY); // convert image to greyscale and store it

  Mat colorHist = Mat::zeros(Size(histsize, histsize), CV_32FC1); // color histogram
  Mat gradientHist = Mat::zeros(1, histsize, CV_32F);             // gradient histogram

  // sobel X and sobel Y for gradient calculation
  Mat sobelX, sobelY;
  Sobel(greyscaleImg, sobelX, CV_32F, 1, 0);
  Sobel(greyscaleImg, sobelY, CV_32F, 0, 1);

  // loop through the sobel image by pixel
  for (int x = 0; x < sobelX.rows; x++)
  {
    for (int y = 0; y < sobelX.cols; y++)
    {
      // store x and y gradients
      float gradientX = sobelX.at<float>(x, y);
      float gradientY = sobelY.at<float>(x, y);

      // compute magnitude
      float magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);

      // map the gradient magnitude to bins
      int bin = static_cast<int>(histsize - 1, histsize * magnitude / 256.0);

      // increment corresponding bin in the histogram
      gradientHist.at<float>(0, bin)++;
    }
  }

  normalize(gradientHist, gradientHist, 1.0, 0.0, NORM_L1); // normalize the gradient histogram

  // store each value in the gradient histogram in the features vector
  for (int i = 0; i < gradientHist.cols; i++)
  {
    features.push_back(gradientHist.at<float>(0, i));
  }

  // loop through each pixel
  for (int i = 0; i < img.rows; i++)
  {
    Vec3b *ptr = img.ptr<Vec3b>(i); // point to row i
    for (int j = 0; j < img.cols; j++)
    {
      // get BGR values
      float B = ptr[j][0]; // blue
      float G = ptr[j][1]; // green
      float R = ptr[j][2]; // red

      // compute r,g chromaticity
      float divisor = B + G + R;
      divisor = divisor > 0 ? divisor : 1.0; // if divisor = 0, assign 1.0 as default

      // normalize r and g based on divisor for weights of each relative to all channels
      float r = R / divisor;
      float g = G / divisor;

      // compute index
      int rindex = (int)(r * (histsize - 1) + 0.5);
      int gindex = (int)(g * (histsize - 1) + 0.5);

      // increment histogram at respective r and g index
      colorHist.at<float>(rindex, gindex)++;
    }
  }

  colorHist /= (img.rows * img.cols); // normalizes all values of the histogram

  // loop through each value of the histogram and store the value in a vector
  for (int i = 0; i < colorHist.rows; i++)
  {
    float *ptr = colorHist.ptr<float>(i);
    for (int j = 0; j < colorHist.cols; j++)
    {
      features.push_back(ptr[j]);
    }
  }
  return 0; // success
}

/**
 * This function creates the feature csv files given the directory of images.
 * dirname - the name of the directory
 * all other arguments - feature CSV file path
 */
int createFeatureCSVFiles(char *dirname, char *feature7x7CSV, char *featureHistCSV,
                          char *featureMultiHistCSV, char *featureColorTextureHistCSV)
{
  // delete the csv files
  if (remove(feature7x7CSV) != 0 && remove(featureHistCSV) != 0 && remove(featureMultiHistCSV) != 0 && remove(featureColorTextureHistCSV) != 0)
  {
    std::cerr << "Error: Failed to delete file: " << feature7x7CSV << endl;
  }

  cout << "File " << feature7x7CSV << " has been deleted." << endl;
  cout << "File " << featureHistCSV << " has been deleted." << endl;
  cout << "File " << featureMultiHistCSV << " has been deleted." << endl;
  cout << "File " << featureColorTextureHistCSV << " has been deleted." << endl;

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
      printf("processing image file: %s\n", dp->d_name);

      // vectors for features
      vector<float> feature7x7vector;
      vector<float> histFeatureVector;
      vector<float> multiHistFeatureVector;
      vector<float> colorTextureHistFeatureVector;

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
      featureHist(currentImg, histFeatureVector);
      append_image_data_csv(featureHistCSV, buffer, histFeatureVector, 0);

      // calculate the multi histogram feature and append it to the csv file
      featureMultiHist(currentImg, multiHistFeatureVector);
      append_image_data_csv(featureMultiHistCSV, buffer, multiHistFeatureVector, 0);

      // calculate the color texture histogram feature and append it to the csv file
      featureColorTextureHist(currentImg, colorTextureHistFeatureVector);
      // append_image_data_csv(featureColorTextureHistCSV, buffer, colorTextureHistFeatureVector, 0);

      // created CSV
      csvCreated = true;
    }
  }
  if (!csvCreated) // failed to create CSV
  {
    return -2;
  }

  return 0;
}