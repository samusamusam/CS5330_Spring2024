/**
 * Sam Lee
 * Spring 2024
 * CS5330
 * 
 * Video Display Application
 *
 * Open a video channel, create a window, and then loop.
 * Within the loop, capture a new frame and display it each time through the loop.
 * On keypress 'q', program quits.
 * On keypress 's', user can save a screenshot as an image file.
 * On keypress 'g', a greyscale version of the video stream is displayed.
 * On keypress 'h', a custom greyscale version of the video stream is displayed.
 * On keypress 'p', a sepia version of the video stream is displayed.
 * On keypress 'b', a blurred version of the video stream is displayed.
 * On keypress 'x', a x sobel version of the video stream is displayed.
 * On keypress 'y', a y sobel version of the video stream is displayed.
 * On keypress 'm', a xy sobel version of the video stream is displayed.
 * On keypress 'l', a quantized version of the video stream is displayed.
 * On keypress 'f', a face tracking version of the video stream is displayed.
 * On keypress 'w', a green version of the video stream is displayed.
 * On keypress 'e', a median filter version of the video stream is displayed.
 * On keypress 'r', a face covering version of the video stream is displayed.
 * One keypress 't', an eye tracker in greyscale activates.
 */

#include "opencv2/opencv.hpp"
#include "filter.h"
#include "faceDetect/faceDetect.h"
#include <cstdio>
#include <cstring>

using namespace cv;

int main(int argc, char *argv[])
{
    // initialize VideoCapture variable as pointer
    VideoCapture *capdev;

    // open the video device
    capdev = new VideoCapture(0);

    // if capdev is not open
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get some properties of the image
    Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH),
              (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // creates and identifies a window
    namedWindow("Video", 1);

    // initialize frame variables
    Mat frame;

    // initialize key press variable
    char key;

    // indefinite for loop that breaks based on key presses
    for (;;)
    {
        // get a new frame from the camera, treat as a stream and store the most recent stream as a frame
        *capdev >> frame;

        // if no frame
        if (frame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // initialize new frame
        Mat newframe = frame;
        std::vector<Rect> faces;
        std::vector<Rect> eyes;
        Mat xsobelimg;
        Mat ysobelimg;

        // filter frame based on current key
        switch (key)
        {
        case 'g': // if 'g' is the current key, show standard greyscale image
            cvtColor(frame, newframe, COLOR_RGB2GRAY);
            break;
        case 'h': // if 'h' is the current key, show custom greyscale image
            customGreyscale(frame, newframe);
            break;
        case 'p': // if 'p' is the current key, show sepia image
            sepiaFilter(frame, newframe);
            break;
        case 'b': // if 'b' is the current key, show blurred image
            blur5x5_2(frame, newframe);
            break;
        case 'x': // if 'x' is the current key, show x sobel image
            sobelX3x3(frame, xsobelimg);
            convertScaleAbs(xsobelimg, newframe);
            break;
        case 'y': // if 'y' is the current key, show y sobel image
            sobelY3x3(frame, ysobelimg);
            convertScaleAbs(ysobelimg, newframe);
            break;
        case 'm': // if 'm' is the current key, show xy sobel image
            sobelX3x3(frame, xsobelimg);
            sobelY3x3(frame, ysobelimg);
            magnitude(xsobelimg, ysobelimg, newframe);
            break;
        case 'l': // if 'l' is the current key, show blur quantized image
            blurQuantize(frame, newframe, 10);
            break;
        case 'f':                                      // if 'f' is the current key, show box around face
            cvtColor(frame, newframe, COLOR_RGB2GRAY); // convert image to greyscale first
            detectFaces(newframe, faces);              // detect faces
            drawBoxes(newframe, faces, 0, 1.0);        // draw boxes around faces
            break;
        case 'w': // if 'w' is the current key, remove RB channels and leave just G channels
            greenify(frame, newframe);
            break;
        case 'e': // if 'e' is the current key, show median filter image
            medianFilter(frame, newframe);
            break;
        case 'r':                                      // if 'r' is the current key, show face covered image
            cvtColor(frame, newframe, COLOR_RGB2GRAY); // convert image to greyscale first
            detectFaces(newframe, faces);              // detect faces
            faceCoverFilter(frame, newframe, faces);
            break;
        case 't':                                      // if 't' is the current key, detect eyes
            cvtColor(frame, newframe, COLOR_RGB2GRAY); // convert image to greyscale first
            detectEyes(newframe, eyes);                // detect eyes and draw circles
            break;
        }

        // see if there is a waiting keystroke
        int keyPressed = waitKey(10);

        // if key pressed, assign to key
        if (keyPressed != -1)
        {
            // print statement based on key pressed
            switch (keyPressed)
            {
            case 'g': // if 'g' is typed, apply standard gresycale filter
                printf("Toggled standard greyscale filter...\n");
                key = static_cast<uchar>(keyPressed);
                break;
            case 'h': // if 'h' is typed, apply custom greyscale filter
                printf("Toggled custom greyscale filter...\n");
                break;
            case 'p': // if 'p' is typed, apply sepia filter
                printf("Toggled sepia filter...\n");
                break;
            case 'b': // if 'b' is typed, apply y sobel filter
                printf("Toggled blur filter...\n");
                break;
            case 'x': // if 'x' is typed, apply x sobel filter
                printf("Toggled x sobel filter...\n");
                break;
            case 'y': // if 'y' is typed, apply gradient sobel filter
                printf("Toggled y sobel filter...\n");
                break;
            case 'm': // if 'm' is typed, apply gradient sobel filter
                printf("Toggled gradient sobel filter...\n");
                break;
            case 'l': // if 'l' is typed, apply blur quantize filter
                printf("Toggled blur quantize filter...\n");
                break;
            case 'f': // if 'f' is typed, detect the face
                printf("Toggled face detect...\n");
                break;
            case 'w': // if 'w' is typed, leave just green channels on image
                printf("Toggled greenify...\n");
                break;
            case 'e': // if 'e' is typed, apply median 3x3 filter
                printf("Toggled median filter...\n");
                break;
            case 'r': // if 'r' is typed, apply face cover filter
                printf("Toggled face covering filter...\n");
                break;
            case 't': // if 't' is typed, detect eye activity
                printf("Toggled eye activity tracker...\n");
                break;
            }

            // assign key the updated keypressed value
            if (keyPressed != 's' && keyPressed != 'q')
            {
                key = static_cast<uchar>(keyPressed);
            }
            // if 'q' is typed, then exit for loop
            if (keyPressed == 'q')
            {
                break;
            }
            // if 's' is  typed, then save screenshot
            if (keyPressed == 's')
            {
                imwrite("screenshot.jpg", newframe);
                printf("Image saved as screenshot.jpg\n");
            }
        }

        // flush the output buffer
        fflush(stdout);

        // show video frame
        imshow("Video", newframe);
    }
    // terminate capdev
    printf("Terminating...\n");
    delete capdev;
    return (0);
}