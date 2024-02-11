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
#include "../faceDetect/faceDetect.h"

using namespace std;
using namespace cv;

/**
 * This function normalizes the feature vector by total bins size
 * features - the feature vector to normalize
 * totalBins - divisor for each feature vector float value
 */
void normalizeFeatures(vector<float> &features, int totalBins)
{
  for (int i = 0; i < features.size(); i++)
  {
    features[i] /= totalBins;
  }
}

/**
 * This function generates the color feature of an image and updates
 * the features vector and the histogram matrix. This uses the rg chromaticity valus.
 * img - the image to get the features for
 * startrow - index of row to start at
 * startcol - index of column to start at
 * endrow - index of row to end at
 * endcol - index of column to end at
 * hist - the histogram that is updated
 * histsize - size of the histogram on each side
 * features - the features of the vector to be stored
 */
int colorHistFeature(Mat &img, int startrow, int startcol, int endrow, int endcol, Mat &hist, int histsize, vector<float> &features)
{
  // loop through each pixel
  for (int i = startrow; i < endrow; i++)
  {
    Vec3b *ptr = img.ptr<Vec3b>(i); // point to row i
    for (int j = startcol; j < endcol; j++)
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

  // hist /= (histsize * histsize); // normalizes all values of the histogram

  // loop through each value of the histogram and store the value in a vector
  for (int i = 0; i < hist.rows; i++)
  {
    float *ptr = hist.ptr<float>(i);
    for (int j = 0; j < hist.cols; j++)
    {
      features.push_back(ptr[j]);
    }
  }

  return 0;
}

/**
 * This function generates the gradient feature of an image and updates
 * the features vector and the histogram matrix
 * greyscaleImg - the image to get the features for
 * hist - the histogram that is updated
 * histsize- size of the histogram on each side
 * features - the features of the vector to be stored
 */
int gradientHistFeature(Mat &greyscaleImg, Mat &hist, int histsize, vector<float> &features)
{
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
      int bin = static_cast<int>(MIN(histsize - 1, histsize * magnitude / 256.0));

      // increment corresponding bin in the histogram
      hist.at<float>(0, bin)++;
    }
  }

  // store each value in the gradient histogram in the features vector
  for (int i = 0; i < hist.cols; i++)
  {
    features.push_back(hist.at<float>(0, i));
  }

  return 0;
}

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

  // d start points as reference for the 7x7 pixel
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
  features.clear();                                          // empty features vector
  int histsize = 16;                                         // bin size
  Mat hist = Mat::zeros(Size(histsize, histsize), CV_32FC1); // 2-D histogram

  colorHistFeature(img, 0, 0, img.rows, img.cols, hist, histsize, features); // update the features based on color histogram

  // normalize features based on histsize
  normalizeFeatures(features, histsize * histsize);

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
  features.clear();                                                    // empty features vector
  int histsize = 16;                                                   // bin size
  Mat topHalfHist = Mat::zeros(Size(histsize, histsize), CV_32FC1);    // top half histogram
  Mat bottomHalfHist = Mat::zeros(Size(histsize, histsize), CV_32FC1); // bottom half histogram

  colorHistFeature(img, 0, 0, int((img.rows / 2)) + 1, int(img.cols), topHalfHist, histsize, features);                   // get color histogram features for top half
  colorHistFeature(img, int((img.rows / 2)), 0, int((img.rows / 2)) + 1, int(img.cols), topHalfHist, histsize, features); // get color histogram features for bottom half
  normalizeFeatures(features, histsize * histsize);

  return 0; // success
}

/**
 * This function calculates the feature vector via a color histogram and a texture histogram
 * img - the image to get the features for
 * features - the features of the vector to be stored
 */
int featureColorTextureHist(Mat &img, vector<float> &features)
{
  features.clear();                                               // empty features vector
  int histsize = 16;                                              // bin size
  Mat colorHist = Mat::zeros(Size(histsize, histsize), CV_32FC1); // color histogram
  Mat gradientHist = Mat::zeros(1, histsize * histsize, CV_32F);             // gradient histogram
  vector<float> colorFeatures;
  vector<float> gradientFeatures;

  colorHistFeature(img, 0, 0, img.rows, img.cols, colorHist, histsize, colorFeatures); // get color histogram features
  normalizeFeatures(colorFeatures, histsize * histsize);

  Mat greyscaleImg;                            // greyscale image
  cvtColor(img, greyscaleImg, COLOR_BGR2GRAY); // convert image to greyscale and store it

  gradientHistFeature(greyscaleImg, gradientHist, histsize*histsize, gradientFeatures); // get gradient histogram features

  normalizeFeatures(gradientFeatures, histsize*histsize);

  // Combine color and gradient features into a single feature vector
  features.insert(features.end(), colorFeatures.begin(), colorFeatures.end());
  features.insert(features.end(), gradientFeatures.begin(), gradientFeatures.end());

  return 0; // success
}

/**
 * This function calculates the feature vector via a color histogram, texture histogram, and DNN
 * img - the image to get the features for
 * features - the features of the vector to be stored
 * imgName - name of the image
 */
int featureColorTextureDNNHist(Mat &img, vector<float> &features, string imgName,
                               vector<char *> &imgFileNames, vector<vector<float>> &imgFeatureData)
{
  features.clear();  // empty features vector
  int histsize = 16; // bin size

  Mat colorHist = Mat::zeros(Size(histsize, histsize), CV_32FC1); // color histogram
  Mat gradientHist = Mat::zeros(1, histsize * histsize, CV_32F);             // gradient histogram

  vector<float> colorFeatures;
  vector<float> gradientFeatures;

  colorHistFeature(img, 0, 0, img.rows, img.cols, colorHist, histsize, colorFeatures); // get color histogram features
  normalizeFeatures(colorFeatures, histsize * histsize);

  Mat greyscaleImg;                            // greyscale image
  cvtColor(img, greyscaleImg, COLOR_BGR2GRAY); // convert image to greyscale and store it

  gradientHistFeature(greyscaleImg, gradientHist, histsize*histsize, gradientFeatures); // get gradient histogram features
  normalizeFeatures(gradientFeatures, histsize*histsize);

  // Combine color and gradient features into a single feature vector
  features.insert(features.end(), colorFeatures.begin(), colorFeatures.end());
  features.insert(features.end(), gradientFeatures.begin(), gradientFeatures.end());

  // push back features from DNN to features
  for (int i = 0; i < imgFeatureData.size(); i++)
  {
    if (imgFileNames[i] == imgName)
    {
       for (int j = 0; j < imgFeatureData[i].size(); j++)
      {
        features.push_back(imgFeatureData[i][j]);
      }
      return 0;
    }
  }

  return 0;
}

/**
 * This function finds all faces in a given image and creates the csv features file
 * img - the image to get the faces for
 * imgFileName - name of the image
 * featureCSV - name of the feature CSV file
 */
int featureFaces(Mat &img, char *imgFileName, char *featureCSV)
{
  int histsize = 256;                                 // bin size
  Mat gradientHist = Mat::zeros(1, histsize, CV_32F); // gradient histogram
  Mat greyscaleImg;                                   // greyscale image
  vector<Rect> faces;
  vector<float> features;

  cvtColor(img, greyscaleImg, COLOR_BGR2GRAY); // convert image to greyscale and store it

  faces.clear();
  detectFaces(greyscaleImg, faces); // get all faces

  // no faces found
  if (faces.size() == 0)
  {
    return -1;
  }

  // for each face found, create a row in the CSV file
  for (int i = 0; i < faces.size(); i++)
  {
    features.clear();
    Rect faceRect = faces[i];
    Mat faceROI = greyscaleImg(faceRect);
    gradientHistFeature(faceROI, gradientHist, histsize, features); // get gradient histogram features of each face
    normalizeFeatures(features, histsize);
    append_image_data_csv(featureCSV, imgFileName, features, 0); // add to CSV
  }

  return 0;
}

/**
 * This function finds the feature vector for the first face found in an image
 * img - the image to get the faces for
 * features - features vector of the face
 */
int featureFirstFace(Mat &img, vector<float> &features)
{
  int histsize = 256;                                 // bin size
  Mat gradientHist = Mat::zeros(1, histsize, CV_32F); // gradient histogram
  Mat greyscaleImg;                                   // greyscale image
  vector<Rect> faces;

  cvtColor(img, greyscaleImg, COLOR_BGR2GRAY); // convert image to greyscale and store it

  faces.clear();
  features.clear();
  detectFaces(greyscaleImg, faces); // get all faces

  // no faces found
  if (faces.size() == 0)
  {
    cerr << "No faces found in image. Please pick another image or use another matching method." << endl;
    return -1;
  }

  // get first face found
  Rect faceRect = faces[0];
  Mat faceROI = greyscaleImg(faceRect);
  gradientHistFeature(faceROI, gradientHist, histsize, features); // get gradient histogram features of first face found
  normalizeFeatures(features, histsize);

  return 0;
}

/**
 * This function creates the feature csv files given the directory of images.
 * dirname - the name of the directory
 * all other arguments - feature CSV file path or pre-loaded csv file data
 */
int createFeatureCSVFiles(char *dirname, char *feature7x7CSV, char *featureHistCSV,
                          char *featureMultiHistCSV, char *featureColorTextureHistCSV,
                          char *featureColorTextureDNNHistCSV, char *featureFaceCSV,
                          vector<char *> &imgFileNames, vector<vector<float>> &imgFeatureData)
{
  // delete the csv files
  if (remove(feature7x7CSV) != 0)
  {
    cerr << "Error: Failed to delete " << feature7x7CSV << " file." << endl;
  }
  if (remove(featureHistCSV) != 0)
  {
    cerr << "Error: Failed to delete " << featureHistCSV << " file." << endl;
  }
  if (remove(featureMultiHistCSV) != 0)
  {
    cerr << "Error: Failed to delete " << featureMultiHistCSV << " file." << endl;
  }
  if (remove(featureColorTextureHistCSV) != 0)
  {
    cerr << "Error: Failed to delete " << featureColorTextureHistCSV << " file." << endl;
  }
  if (remove(featureColorTextureDNNHistCSV) != 0)
  {
    cerr << "Error: Failed to delete " << featureColorTextureDNNHistCSV << " file." << endl;
  }
  if (remove(featureFaceCSV) != 0)
  {
    cerr << "Error: Failed to delete " << featureFaceCSV << " file." << endl;
  }

  cout << "File " << feature7x7CSV << " has been deleted." << endl;
  cout << "File " << featureHistCSV << " has been deleted." << endl;
  cout << "File " << featureMultiHistCSV << " has been deleted." << endl;
  cout << "File " << featureColorTextureHistCSV << " has been deleted." << endl;
  cout << "File " << featureColorTextureDNNHistCSV << " has been deleted." << endl;
  cout << "File " << featureFaceCSV << " has been deleted." << endl;

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
      // printf("processing image file: %s\n", dp->d_name);

      // vectors for features
      vector<float> feature7x7vector;
      vector<float> histFeatureVector;
      vector<float> multiHistFeatureVector;
      vector<float> colorTextureHistFeatureVector;
      vector<float> colorTextureDNNHistFeatureVector;

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      // print full file path
      // printf("full path name: %s\n", buffer);

      // read current directory image
      Mat currentImg = imread(buffer);

      // calculate the 7x7 feature and append it to the csv file
      feature7x7(currentImg, feature7x7vector);
      append_image_data_csv(feature7x7CSV, dp->d_name, feature7x7vector, 0);

      // calculate the histogram feature and append it to the csv file
      featureHist(currentImg, histFeatureVector);
      append_image_data_csv(featureHistCSV, dp->d_name, histFeatureVector, 0);

      // calculate the multi histogram feature and append it to the csv file
      featureMultiHist(currentImg, multiHistFeatureVector);
      append_image_data_csv(featureMultiHistCSV, dp->d_name, multiHistFeatureVector, 0);

      // calculate the color texture histogram feature and append it to the csv file
      featureColorTextureHist(currentImg, colorTextureHistFeatureVector);
      append_image_data_csv(featureColorTextureHistCSV, dp->d_name, colorTextureHistFeatureVector, 0);

      // calculate the color texture DNN histogram feature and append it to the csv file
      featureColorTextureDNNHist(currentImg, colorTextureDNNHistFeatureVector, dp->d_name, imgFileNames, imgFeatureData);
      append_image_data_csv(featureColorTextureDNNHistCSV, dp->d_name, colorTextureDNNHistFeatureVector, 0);

      // calculate the face feature csv file
      featureFaces(currentImg, dp->d_name, featureFaceCSV);

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
