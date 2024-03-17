/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This file includes functions that can detect corners of an image with a chessboard
 */

#include "opencv2/opencv.hpp"

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