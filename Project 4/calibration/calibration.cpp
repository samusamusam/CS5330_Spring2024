/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This file includes functions used for calibrating an image
 */

#include "opencv2/opencv.hpp"
#include <vector>

using namespace std;
using namespace cv;

/**
 * This function gets the world points of a chessboard according to its size
 * point_set - data structure to store world points in
 * chessboardSize - size of the chessboard
*/
int getChessboardWorldPoints(vector<Point3f> &point_set, const Size &chessboardSize)
{
  for (int i = 0; i < chessboardSize.height; i++)
  {
    for (int j = 0; j < chessboardSize.width; j++)
    {
      point_set.push_back(Vec3f(j, -i, 0));
    }
  }
  return (0);
}

/**
 * This function projects the 3D axes onto the target in the image plane
 * rotations - matrix that represents rotation of target
 * translations - matrix that represents translation of target
 * cameraMat - matrix that represents the camera
 * distortionCoeffs - matrix of distortion coefficients
 * img - image to project the axes onto
*/
int project3DAxes(Mat &rotations, Mat &translations, Mat &cameraMat, Mat &distortionCoeffs, Mat &img) {
  // create axis points of the 3D axes
  vector<Point3f> axisPoints;
  axisPoints.push_back(Point3f(0,0,0)); // origin
  axisPoints.push_back(Point3f(3,0,0)); // X axis
  axisPoints.push_back(Point3f(0,3,0)); // Y axis
  axisPoints.push_back(Point3f(0,0,3)); // Z axis

  // get image points 
  vector<Point2f> imagePoints;
  projectPoints(axisPoints, rotations, translations, cameraMat, distortionCoeffs, imagePoints);

  // draw the 3D axes
  line(img, imagePoints[0], imagePoints[1], Scalar(255,0,0), 12); // blue = x-axis
  line(img, imagePoints[0], imagePoints[2], Scalar(0,255,0), 12); // green = y-axis
  line(img, imagePoints[0], imagePoints[3], Scalar(0,0,255), 12); // red = z-axis

  return 0;
}

/**
 * This function projects a 3D shape onto the target in the image plane
 * rotations - matrix that represents rotation of target
 * translations - matrix that represents translation of target
 * cameraMat - matrix that represents the camera
 * distortionCoeffs - matrix of distortion coefficients
 * img - image to project the shape onto
*/
int projectShape3D(Mat &rotations, Mat &translations, Mat &cameraMat, Mat &distortionCoeffs, Mat &img) {
  // create objects points of the 3D image
  vector<Point3f> objectPoints;
  // bottom face
  objectPoints.push_back(Point3f(1,-1,1));
  objectPoints.push_back(Point3f(4,-1,1));
  objectPoints.push_back(Point3f(1,-3,1));
  objectPoints.push_back(Point3f(4,-3,1));
  // top face
  objectPoints.push_back(Point3f(4.5,-1,4));
  objectPoints.push_back(Point3f(7.5,-1,4));
  objectPoints.push_back(Point3f(4.5,-3,4));
  objectPoints.push_back(Point3f(7.5,-3,4));
  
  // get image points 
  vector<Point2f> imagePoints;
  projectPoints(objectPoints, rotations, translations, cameraMat, distortionCoeffs, imagePoints);

  // draw the 3D shape
  line(img, imagePoints[0], imagePoints[1], Scalar(255,0,0), 8);
  line(img, imagePoints[0], imagePoints[2], Scalar(255,0,0), 8);
  line(img, imagePoints[3], imagePoints[2], Scalar(255,0,0), 8);
  line(img, imagePoints[3], imagePoints[1], Scalar(255,0,0), 8);

  line(img, imagePoints[0], imagePoints[4], Scalar(255,0,0), 8);
  line(img, imagePoints[1], imagePoints[5], Scalar(255,0,0), 8);
  line(img, imagePoints[2], imagePoints[6], Scalar(255,0,0), 8);
  line(img, imagePoints[3], imagePoints[7], Scalar(255,0,0), 8);

  line(img, imagePoints[4], imagePoints[5], Scalar(255,0,0), 8);
  line(img, imagePoints[4], imagePoints[6], Scalar(255,0,0), 8);
  line(img, imagePoints[7], imagePoints[6], Scalar(255,0,0), 8);
  line(img, imagePoints[7], imagePoints[5], Scalar(255,0,0), 8);

  return 0;
}