/**
 * Samuel Lee
 * Spring 2024
 * CS 5330 Tutorial 2
*/

#include <cstdio>
#include <cstring>
#include <sys/time.h>
#include "opencv2/opencv.hpp"
#include "Gaussian.h"

double getTime() {
    struct timeval cur;

    gettimeofday( &cur, NULL );
    return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}

// reading an image (path on command line) and modifying it
int main(int argc, char *argv[]) {
    cv::Mat src; // standard image datatype
    cv::Mat dst; // holds the results
    char filename[256];

    // check if enough command line arguments
    if(argc < 2) {
        printf("usage: %s <image filename>\n", argv[0]);
        exit(-1);
    }

    strcpy( filename, argv[1] );

    // read the image
    src = cv::imread( filename ); // allocates the image data, reads as a BGR 8-bit per channel image

    if ( src.data == NULL) { // no image data read from file
        printf("error: unable to read image %s\n", filename );
        exit(-1);
    }

    cv::namedWindow(filename,1);
    cv::imshow(filename,src);

    // call blur 3 times
    double start = getTime();
    gauss3x3ptr( src, dst);
    gauss3x3ptr( dst, src);
    gauss3x3ptr( src, dst);
    double end = getTime();
    printf("Time for ptr<> method: %.5f\n", (end - start) / 3);

    // call blur 3 times
    start = getTime();
    gauss3x3at( src, dst);
    gauss3x3at( dst, src);
    gauss3x3at( src, dst);
    end = getTime();
    printf("Time for at<> method: %.5f\n", (end - start) / 3);

    cv::namedWindow( "Blur", 1 );
    cv::imshow( "Blur", src ); 

    cv::waitKey(0); // waits for a keypress

    cv::destroyWindow( filename );

    printf("Terminating\n");

    return(0);

}