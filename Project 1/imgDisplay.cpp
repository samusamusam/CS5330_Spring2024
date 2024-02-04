/**
 * Sam Lee
 * Spring 2024
 * CS 5330
 * 
 * Image Display Application
 * 
 * Opens an image and reads keypress
 * 'q' = close window
 * 'c' = copy and create a new image
*/

#include "opencv2/opencv.hpp"
#include <cstdio>
#include <cstring>

using namespace cv;

int main(int argc, char *argv[]) {
    // set image and filename variables
    Mat img; 
    char filename[256];

    // check if enough command line arguments
    if(argc < 2) {
        printf("usage: %s <image filename>\n", argv[0]);
        exit(-1);
    }

    // copy argv[1] to the filename variable
    strcpy( filename, argv[1] );

    // read the image
    img = imread( filename );

    // if no image data read from file
    if ( img.data == NULL) {
        printf("error: unable to read image %s\n", filename );
        exit(-1);
    }

    // open window of image
    namedWindow( filename, 1 );
    imshow( filename, img );

    // get key pressed
    int key = waitKey(0);

    // if key does not equal q, keep updating key
    while(key != 'q') {
        // if 'c' is pressed, we will create a copy of this image in the current directory
        if(key == 'c') {
            printf("Creating a copy of the image\n");
            imwrite("copiedimage.jpg", img);
            printf("Copy of image created with name copiedimage.jpg\n");
        }
        key = waitKey(0);
    }

    // wait for 'q' to be pressed for quitting
    destroyWindow( filename ); 
    printf("Terminating\n");   

    return(0);
}
