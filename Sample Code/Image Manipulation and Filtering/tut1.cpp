/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Tutorial 1

  Opening and manipulating an image
*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"  // the top include file

// reading an image (path on command line), modifying it
int main( int argc, char *argv[] ) {
  cv::Mat src; // standard image data type
  char filename[256];

  // check if enough command line arguments
  if(argc < 2) {
    printf("usage: %s <image filename>\n", argv[0] );
    exit(-1);
  }
  strcpy( filename, argv[1] );

  // read the image
  src = cv::imread( filename ); // allocates the image data, reads as a BGR 8-bit per channel image

  // test of the image read was successful
  if( src.data == NULL ) { // no image data read from file
    printf("error: unable to read image %s\n", filename );
    exit(-1);
  }

  // successfully read an image file
  printf("Image size:        %d rows %d columns\n", (int)src.size().height, (int)src.size().width );
  // src.rows, src.cols also works for height (rows) and width (cols)
  printf("Number of channel: %d\n", (int)src.channels() );
  printf("Bytes per channel: %d\n", (int)src.elemSize() / src.channels() );

  cv::namedWindow( filename, 1 );
  cv::imshow( filename, src ); // src must be a BGR 8-bit per channel image

  // Let's modify the image to swap the red and green channels
  /*
  // uses the at<> method
  for(int i=0;i<src.rows;i++) {
    for(int j=0;j<src.cols;j++) {
      uchar tmp = src.at<cv::Vec3b>(i,j)[1]; // save the green channel to tmp
      src.at<cv::Vec3b>(i,j)[1] = src.at<cv::Vec3b>(i,j)[2];
      src.at<cv::Vec3b>(i,j)[2] = tmp;
    }
  }
  */

  // uses the ptr<> method
  for(int i=0;i<src.rows;i++) {
    cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i); // get a pointer to row i
    for(int j=0;j<src.cols;j++) {
      uchar tmp = ptr[j][1]; // saving the green value
      ptr[j][1] = ptr[j][2];
      ptr[j][2] = tmp;
    }
  }
  

  cv::namedWindow( "Swap RG", 2 );
  cv::imshow( "Swap RG", src );

  cv::waitKey(0); // waits for a keypress

  cv::destroyWindow( filename );

  printf("Terminating\n");

  return(0);
}
  
