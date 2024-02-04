/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Implementation of various simple filters
*/

#include <cstdio>
#include "opencv2/opencv.hpp"

/*
  Implement a simple Gaussian 3x3 filter as
  
  1 2 1
  2 4 2
  1 2 1

  This version uses the at<> method of cv::Mat
*/
int gauss3x3at( cv::Mat &src, cv::Mat &dst ) { // pass images by reference

  // allocate space for the dst image
  src.copyTo( dst ); // makes a copy of the original data

  // alternative
  // dst.create( src.size(), src.type() ); // makes a duplicate size image, uninitalized data

  // only going to calculate values for pixels where filter fits in the image
  // nested loop
  for(int i=1;i<src.rows-1;i++) { // rows
    for(int j=1;j<src.cols-1;j++) { // columns
      for(int k=0;k<src.channels();k++) { // color channels
	int sum = src.at<cv::Vec3b>(i-1, j-1)[k] + 2 * src.at<cv::Vec3b>(i-1, j)[k] + src.at<cv::Vec3b>(i-1, j+1)[k] +
	  2 * src.at<cv::Vec3b>(i, j-1)[k] + 4 * src.at<cv::Vec3b>(i, j)[k] + 2 * src.at<cv::Vec3b>(i, j+1)[k] +
	  src.at<cv::Vec3b>(i+1, j-1)[k] + 2 * src.at<cv::Vec3b>(i+1, j)[k] + src.at<cv::Vec3b>(i+1, j+1)[k];

	// normalize the value back to a range of [0, 255]
	sum /= 16;

	dst.at<cv::Vec3b>(i, j)[k] = sum;
      }
    }
  }
  // done

  return(0);
}


/*
  Implement a 3x3 Gaussian using the ptr<> method
*/
int gauss3x3ptr( cv::Mat &src, cv::Mat &dst ) { // pass images by reference

  src.copyTo(dst); // allocate dst cv::Mat

  for(int i=1;i<src.rows-1;i++) { // rows
    cv::Vec3b *ptrup = src.ptr<cv::Vec3b>(i-1); // data for row i-1
    cv::Vec3b *ptrmd = src.ptr<cv::Vec3b>(i); // pointer to the data for row i
    cv::Vec3b *ptrdn = src.ptr<cv::Vec3b>(i+1); // data for row i+1
    cv::Vec3b *dptr  = dst.ptr<cv::Vec3b>(i); // result image data for row i

    for(int j=1;j<src.cols-1;j++) { // cols
      for(int k=0;k<src.channels();k++) {

	// ptrmd[j][k] accesses position (i, j) channel k
	int sum = ptrup[j-1][k] + 2 * ptrup[j][k] + ptrup[j+1][k] +
	  2 * ptrmd[j-1][k] + 4 * ptrmd[j][k] + 2 * ptrmd[j+1][k] +
	  ptrdn[j-1][k] + 2 * ptrdn[j][k] + ptrdn[j+1][k];

	sum /= 16;

	dptr[j][k] = sum; // assign the result using dst ptr
      }
    }
  }
  // done

  return(0);
}
