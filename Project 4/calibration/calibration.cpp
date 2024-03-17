/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This file includes functions used for calibrating an image
 */

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**
 * This function gets the world points of a chessboard according to its size
 * point_set - data structure to store world points in
 * chessboardSize - size of the chessboard
*/
int getChessboardWorldPoints(vector<Vec3f> &point_set, const Size &chessboardSize)
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