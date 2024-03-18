/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This file includes functions that can detect corners of an image with a chessboard
 */

#include "opencv2/opencv.hpp"
#include <vector>

using namespace std;
using namespace cv;

/**
 * This function gets the corners of the chessboard shown in an image
 * chessboardSize - size of the chessboard
 * cornerSet - corners found in the chessboard
 * image - image to detect chessboard from
 * cornersFound - boolean value indicating whether the corners were found
 */
int getCorners(const Size chessboardSize, vector<Point2f> &cornerSet, Mat &image, bool &cornersFound)
{

  // initialize gray image
  Mat grayImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);

  // find corners of chessboard
  cornersFound = findChessboardCorners(grayImage, chessboardSize, cornerSet, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FILTER_QUADS + CALIB_CB_FAST_CHECK);

  // if corners found successfully
  if (cornersFound)
  {
    // variables for cornerSubPix
    Size searchArea(11, 11);
    Size zeroZone(-1, -1);
    TermCriteria terminationCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);

    // refine position of corners
    cornerSubPix(grayImage, cornerSet, searchArea, zeroZone, terminationCriteria);

    // draw corners
    drawChessboardCorners(image, chessboardSize, cornerSet, cornersFound);
  }
  return (0);
}

/**
 * This function gets the corners of multiple targets in the image
 * chessboardSize - size of the chessboard
 * cornerSet - set of corners of chessboards found in the image
 * image - image to detect chessboard from
 */
int getMultipleCorners(const Size chessboardSize, vector<vector<Point2f>> &multipleCornerSet, Mat &image)
{

  // initialize gray image
  Mat grayImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);

  // initialize corners variables for one target
  vector<Point2f> cornerSet;
  bool cornersFound;

  // initialize mask of image
  Mat mask = Mat::ones(grayImage.size(), CV_8UC1) * 255;

  while (true)
  {
    cornerSet.clear();    // clear cornerSet before each iteration
    cornersFound = false; // set corners found to false before each iteration

    // apply mask to grayscale image
    Mat maskedGray;
    grayImage.copyTo(maskedGray, mask);

    // find corners of chessboard
    cornersFound = findChessboardCorners(maskedGray, chessboardSize, cornerSet, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FILTER_QUADS + CALIB_CB_FAST_CHECK);
    
    // if corners found successfully
    if (cornersFound)
    {
      // variables for cornerSubPix
      Size searchArea(11, 11);
      Size zeroZone(-1, -1);
      TermCriteria terminationCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1);

      // refine position of corners
      cornerSubPix(maskedGray, cornerSet, searchArea, zeroZone, terminationCriteria);

      // add corners to list of all corner sets
      multipleCornerSet.push_back(cornerSet);

      // draw the detected chessboard
      drawChessboardCorners(image, chessboardSize, cornerSet, cornersFound);

      // mask out the currently found corner set to prevent it from being found again
      for (const auto &pt : cornerSet) {
        circle(mask, pt, 10, Scalar(0), -1);
      }
    } else {
      break;
    }
    }
    return (0);
  }