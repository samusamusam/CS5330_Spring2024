/**
 * Samuel Lee
 * CS 5330
 * Spring 2024
 * This file includes functions used for calibrating an image
 */

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int getChessboardWorldPoints(Vector<Vec3f> &point_set, const Size &chessboardSize) {
  for(int i = 0; i < chessboardSize.height; i++) {
    for(int j = 0; j < chessboardSize.width; j++) {
      point_set.push_back(Vec3f(j,-i,0));
    }
  }


}