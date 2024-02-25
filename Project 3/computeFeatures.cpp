/*
  Samuel Lee
  Anjith Prakash Chathan Kandy
  2/22/24
  This file contains functions that compute features of the region identified in an image.
*/

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/**
 * This function calculates the percent filled and height-width ratio features of a region within an image, identified via the region map and region ID.
 * regionMap - region map of the image
 * regionID - ID of the region to read
 * percentFilled - feature #1; percent filles within the bounding box
 * heightWidthRatio - feature #2; bounding box height and width ratio
 * src - image to manipulate directly
 */
int computeFeatures(const Mat &regionMap, int regionID, double &percentFilled, double &heightWidthRatio, Mat &src, Mat &original)
{
  // create a binary mask for the region based on the regionID in the regionMap
  Mat regionMask = (regionMap == regionID);

  // check if region is empty
  if (countNonZero(regionMask) == 0)
  {
    cerr << "Error: Empty region." << endl;
    return -1;
  }

  // find non-zero pixels to calculate bounding box
  vector<Point> nonZeroPoints;
  findNonZero(regionMask, nonZeroPoints);
  Rect boundingBox = boundingRect(nonZeroPoints);

  // calculate percent filled feature
  double area = countNonZero(regionMask);
  percentFilled = (area / (boundingBox.height * boundingBox.width));

  // calculate height:width ratio feature
  heightWidthRatio = static_cast<double>(boundingBox.height) / boundingBox.width;

  // find minimum bounding rectangle to get angle
  RotatedRect rotatedRect = minAreaRect(nonZeroPoints);
  double thetaOrientation = rotatedRect.angle; // in degrees

  // calculate the moments of the region
  Moments moments = cv::moments(regionMask, true);

  // axis of least central moments
  double theta = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

  // draw bounding box and angle indicator
  Point2f vertices[4];
  rotatedRect.points(vertices);
  for (int i = 0; i < 4; ++i)
  {
    line(original, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 8);
  }

  // draw line for axis of least central moments
  Point2f centroid = rotatedRect.center;
  double cosTheta = cos(theta);
  double sinTheta = sin(theta);
  double minSide = min(boundingBox.width, boundingBox.height);
  Point p2(centroid.x + (minSide * cosTheta)/2, centroid.y + (minSide * sinTheta)/2);
  line(original, centroid, p2, Scalar(255, 255, 255), 6);

  // text indicating theta (axis of least central moments)
  stringstream ss;
  ss << "Theta (degrees): " << theta * 180 / CV_PI;
  putText(original, ss.str(), Point(boundingBox.x, boundingBox.y - 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 4);

  // text indicating percent filled feature
  stringstream sspf;
  sspf << "Percent filled: " << percentFilled;
  putText(original, sspf.str(), Point(boundingBox.x, boundingBox.y + boundingBox.height + 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 4);

  return 0;
}
