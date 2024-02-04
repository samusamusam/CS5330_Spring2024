/**
 * Samuel Lee
 * Spring 2024
 * CS 5330
 * Code that contains methods for filtering
 */

#include "opencv2/opencv.hpp"
#include <cstdio>
#include <cstring>

using namespace cv;
using namespace std;

// constants used for custom grey scale RGB weights
const double BLUE_WEIGHT = 0.2;
const double GREEN_WEIGHT = 0.6;
const double RED_WEIGHT = 0.2;

/**
 * This function takes in a src and outputs an image dst with a custom greyscale filter.
 * The filter uses a weighted sum of the RGB values, overemphasizing the green value
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int customGreyscale(Mat &src, Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    // create 8 bit unsigned char single channel destination matrix
    dst.create(src.size(), CV_8UC1);

    // loop through each pixel by row
    for (int i = 0; i < src.rows; i++)
    {
        // get ptr to the row and uchar of dst
        Vec3b *ptr = src.ptr<Vec3b>(i);
        uchar *dstptr = dst.ptr<uchar>(i);

        // loop through each column
        for (int j = 0; j < src.cols; j++)
        {
            // get rgb value at i,j
            Vec3b rgb = ptr[j];
            uchar greyValue = static_cast<uchar>((RED_WEIGHT * rgb[2]) + (GREEN_WEIGHT * rgb[1]) + (BLUE_WEIGHT * rgb[0]));
            // assign the uchar greyvalue to the dst at i,j
            dstptr[j] = greyValue;
        }
    }
    return (0);
}

/**
 * This function takes in a src and outputs an image dst with a sepia filter.
 * The filter uses specific BGR weights corresponding to the sepia filter
 *
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int sepiaFilter(Mat &src, Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    // create 8 bit unsigned char 3 channel destination matrix
    dst.create(src.size(), CV_8UC3);

    // loop through each pixel by row
    for (int i = 0; i < src.rows; i++)
    {
        // get ptr to the row and uchar of dst
        Vec3b *ptr = src.ptr<Vec3b>(i);
        Vec3b *dstptr = dst.ptr<Vec3b>(i);

        // loop through each column
        for (int j = 0; j < src.cols; j++)
        {
            // get rgb value at i,j
            Vec3b rgb = ptr[j];

            dstptr[j][0] = saturate_cast<uchar>((0.131 * rgb[0]) + (0.534 * rgb[1]) + (0.272 * rgb[2])); // blue value
            dstptr[j][1] = saturate_cast<uchar>((0.168 * rgb[0]) + (0.686 * rgb[1]) + (0.349 * rgb[2])); // green value
            dstptr[j][2] = saturate_cast<uchar>((0.189 * rgb[0]) + (0.769 * rgb[1]) + (0.393 * rgb[2])); // red value
        }
    }

    return (0);
}

/**
 * This function takes in a src and outputs an image dst with a 5x5 blur filter.
 * This function utilizes the at<> method
 * The Gaussian filter used is:
 * 1 2 4 2 1
 * 2 4 8 4 2
 * 4 8 16 8 4
 * 2 4 8 4 2
 * 1 2 4 2 1
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    // create 8 bit unsigned char 3 channel destination matrix
    dst.create(src.size(), CV_8UC3);

    // loop through each pixel by row
    for (int i = 2; i < src.rows - 2; i++)
    {
        // loop through each column
        for (int j = 2; j < src.cols - 2; j++)
        {
            // loop through each channel
            for (int k = 0; k < src.channels(); k++)
            {
                // add all pixel values
                int sum = 1 * src.at<Vec3b>(i - 2, j - 2)[k] + 2 * src.at<Vec3b>(i - 2, j - 1)[k] + 4 * src.at<Vec3b>(i - 2, j)[k] + 2 * src.at<Vec3b>(i - 2, j + 1)[k] + 1 * src.at<Vec3b>(i - 2, j + 2)[k] + 2 * src.at<Vec3b>(i - 1, j - 2)[k] + 4 * src.at<Vec3b>(i - 1, j - 1)[k] + 8 * src.at<Vec3b>(i - 1, j)[k] + 4 * src.at<Vec3b>(i - 1, j + 1)[k] + 2 * src.at<Vec3b>(i - 1, j + 2)[k] + 4 * src.at<Vec3b>(i, j - 2)[k] + 8 * src.at<Vec3b>(i, j - 1)[k] + 16 * src.at<Vec3b>(i, j)[k] + 8 * src.at<Vec3b>(i, j + 1)[k] + 4 * src.at<Vec3b>(i, j + 2)[k] + 2 * src.at<Vec3b>(i + 1, j - 2)[k] + 4 * src.at<Vec3b>(i + 1, j - 1)[k] + 8 * src.at<Vec3b>(i + 1, j)[k] + 4 * src.at<Vec3b>(i + 1, j + 1)[k] + 2 * src.at<Vec3b>(i + 1, j + 2)[k] + 1 * src.at<Vec3b>(i + 2, j - 2)[k] + 2 * src.at<Vec3b>(i + 2, j - 1)[k] + 4 * src.at<Vec3b>(i + 2, j)[k] + 2 * src.at<Vec3b>(i + 2, j + 1)[k] + 1 * src.at<Vec3b>(i + 2, j + 2)[k];

                sum /= 100; // for weighted sum, add all the numbers in the filter

                dst.at<Vec3b>(i, j)[k] = sum; // set the destination image value at pixel
            }
        }
    }
    return (0);
}
/**
 * This function takes in a src and outputs an image dst with a 5x5 blur filter.
 * This function utilizes the ptr<> method
 * The Gaussian filter used is:
 * 1 2 4 2 1
 * 2 4 8 4 2
 * 4 8 16 8 4
 * 2 4 8 4 2
 * 1 2 4 2 1
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    Mat qtr; // image quarter of the size initialization
    resize(src, qtr, Size(src.cols / 4, src.rows / 4)); // resize image to quarter of the size

    // create 8 bit unsigned char 3 channel destination matrix
    dst.create(qtr.size(), CV_8UC3);
    // store temporary frame
    Mat temp;
    temp.create(qtr.size(), CV_8UC3);

    // first apply horizontal filter
    // loop through each pixel by row
    for (int i = 2; i < qtr.rows - 2; i++)
    {
        // get pointers
        Vec3b *ptrmd = qtr.ptr<Vec3b>(i);
        Vec3b *dptr = temp.ptr<Vec3b>(i);
        // loop through each column
        for (int j = 2; j < qtr.cols - 2; j++)
        {
            // loop through each channel
            for (int k = 0; k < qtr.channels(); k++)
            {
                // apply filter
                int sum = 1 * ptrmd[j - 2][k] + 2 * ptrmd[j - 1][k] + 4 * ptrmd[j][k] + 2 * ptrmd[j + 1][k] + 1 * ptrmd[j + 2][k];
                sum /= 10; // for weighted sum, add all the numbers in the filter

                dptr[j][k] = sum; // assign value to destination pointer
            }
        }
    }

    // next apply vertical filter on temp
    // loop through each pixel by column
    for (int j = 2; j < temp.cols - 2; j++)
    {
        // loop through each row
        for (int i = 2; i < temp.rows - 2; i++)
        {
            // get pointers
            Vec3b *ptruptwo = temp.ptr<Vec3b>(i - 2);
            Vec3b *ptrupone = temp.ptr<Vec3b>(i - 1);
            Vec3b *ptrmd = temp.ptr<Vec3b>(i);
            Vec3b *ptrdnone = temp.ptr<Vec3b>(i + 1);
            Vec3b *ptrdntwo = temp.ptr<Vec3b>(i + 2);
            Vec3b *dptr = dst.ptr<Vec3b>(i);

            // loop through each channel
            for (int k = 0; k < temp.channels(); k++)
            {
                // apply filter
                int sum = 1 * ptruptwo[j][k] + 2 * ptrupone[j][k] + 4 * ptrmd[j][k] + 2 * ptrdnone[j][k] + 1 * ptrdntwo[j][k];
                sum /= 10; // for weighted sum, add all the numbers in the filter

                dptr[j][k] = sum; // assign value to destination pointer
            }
        }
    }
    // resize back to original size
    resize(dst, dst, cv::Size(src.cols, src.rows));
    return (0);
}

/**
 * This function takes in a src and outputs an image dst with a 3x3 horizontal sobel filter.
 * The filter used is:
 * -1 0 1
 * -2 0 2
 * -1 0 1
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    // create 16 bit signed char 3 channel destination matrix
    dst.create(src.size(), CV_16SC3);

    // store temporary frame
    Mat temp;
    temp.create(src.size(), CV_16SC3);

    // first apply horizontal filter
    // loop through each pixel by row
    for (int i = 1; i < src.rows - 1; i++)
    {
        // get pointers
        Vec3b *ptrmd = src.ptr<Vec3b>(i);
        Vec3s *dptr = temp.ptr<Vec3s>(i);
        // loop through each column
        for (int j = 1; j < src.cols - 1; j++)
        {
            // loop through each channel
            for (int k = 0; k < src.channels(); k++)
            {
                // apply filter
                int sum = -1 * ptrmd[j - 1][k] + 0 * ptrmd[j][k] + 1 * ptrmd[j + 1][k];
                dptr[j][k] = sum; // assign value to destination pointer
            }
        }
    }

    // next apply vertical filter on temp
    // loop through each pixel by column
    for (int j = 1; j < temp.cols - 1; j++)
    {
        // loop through each row
        for (int i = 1; i < temp.rows - 1; i++)
        {
            // get pointers
            Vec3s *ptrupone = temp.ptr<Vec3s>(i - 1);
            Vec3s *ptrmd = temp.ptr<Vec3s>(i);
            Vec3s *ptrdnone = temp.ptr<Vec3s>(i + 1);
            Vec3s *dptr = dst.ptr<Vec3s>(i);

            // loop through each channel
            for (int k = 0; k < temp.channels(); k++)
            {
                // apply filter
                int sum = 1 * ptrupone[j][k] + 2 * ptrmd[j][k] + 1 * ptrdnone[j][k];
                dptr[j][k] = sum; // assign value to destination pointer
            }
        }
    }
    return (0);
}

/**
 * This function take ins a src and outputs an image dst with a 3x3 vertical sobel filter
 * The filter used is:
 * 1 2 1
 * 0 0 0
 * -1 -2 -1
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    // create 16 bit signed char 3 channel destination matrix
    dst.create(src.size(), CV_16SC3);

    // store temporary frame
    Mat temp;
    temp.create(src.size(), CV_16SC3);

    // first apply horizontal filter
    // loop through each pixel by row
    for (int i = 1; i < src.rows - 1; i++)
    {
        // get pointers
        Vec3b *ptrmd = src.ptr<Vec3b>(i);
        Vec3s *dptr = temp.ptr<Vec3s>(i);
        // loop through each column
        for (int j = 1; j < src.cols - 1; j++)
        {
            // loop through each channel
            for (int k = 0; k < src.channels(); k++)
            {
                // apply filter
                int sum = 1 * ptrmd[j - 1][k] + 2 * ptrmd[j][k] + 1 * ptrmd[j + 1][k];
                dptr[j][k] = sum; // assign value to destination pointer
            }
        }
    }

    // next apply vertical filter on temp
    // loop through each pixel by column
    for (int j = 1; j < temp.cols - 1; j++)
    {
        // loop through each row
        for (int i = 1; i < temp.rows - 1; i++)
        {
            // get pointers
            Vec3s *ptrupone = temp.ptr<Vec3s>(i - 1);
            Vec3s *ptrmd = temp.ptr<Vec3s>(i);
            Vec3s *ptrdnone = temp.ptr<Vec3s>(i + 1);
            Vec3s *dptr = dst.ptr<Vec3s>(i);

            // loop through each channel
            for (int k = 0; k < temp.channels(); k++)
            {
                // apply filter
                int sum = 1 * ptrupone[j][k] + 0 * ptrmd[j][k] + -1 * ptrdnone[j][k];
                dptr[j][k] = sum; // assign value to destination pointer
            }
        }
    }
    return (0);
}

/**
 * This function gets the magnitudes of the xy sobel filters
 * Arguments:
 * cv::Mat sx - a signed short image resulting from an x sobel filter
 * cv::Mat sy - a signed short image resulting from a y sobel filter
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    // check if sx or sy is empty
    if (sx.empty() || sy.empty())
    {
        printf("No valid sx or sy file\n");
        return -1;
    }

    // create 8 bit unsigned char 3 channel destination matrix
    dst.create(sx.size(), CV_8UC3);

    // loop through each pixel by row
    for (int i = 0; i < sx.rows; i++)
    {
        // loop through each column
        for (int j = 0; j < sx.cols; j++)
        {
            Vec3s x = sx.at<Vec3s>(i, j); // get x vector
            Vec3s y = sy.at<Vec3s>(i, j); // get y vector

            // find magnitude at each channel
            uchar magnitude_0 = static_cast<uchar>(sqrt(x[0] * x[0] + y[0] * y[0]));
            uchar magnitude_1 = static_cast<uchar>(sqrt(x[1] * x[1] + y[1] * y[1]));
            uchar magnitude_2 = static_cast<uchar>(sqrt(x[2] * x[2] + y[2] * y[2]));

            // assign magnitudes to destination
            dst.at<Vec3b>(i, j)[0] = magnitude_0;
            dst.at<Vec3b>(i, j)[1] = magnitude_1;
            dst.at<Vec3b>(i, j)[2] = magnitude_2;
        }
    }
    return (0);
}

/**
 * This function blurs and quantizes the image by taking in the src and creating the new image in dst
 * The input 'levels' determines how much the image gets quantized
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 * int levels - determines how many values a color channel is quantized to
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    blur5x5_2(src, dst); // blur the image

    int bucketSize = static_cast<int>(255.0 / levels); // determine bucket size

    // loop through each row
    for (int i = 0; i < dst.rows; i++)
    {
        // loop through each column
        for (int j = 0; j < dst.cols; j++)
        {
            // loop through each channel
            for (int c = 0; c < dst.channels(); c++)
            {
                // get new pixel values based on bucket/level
                double xt = static_cast<double>(dst.at<Vec3b>(i, j)[c] / bucketSize);
                dst.at<Vec3b>(i, j)[c] = static_cast<uchar>(xt * bucketSize);
            }
        }
    }
    return (0);
}

/**
 * This function removes red and blue channels and leaves just the green channel
 * Arguments:
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int greenify(cv::Mat &src, cv::Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    // create 8 bit unsigned char 3 channel destination matrix
    dst.create(src.size(), CV_8UC3);

    // loop through each row
    for (int i = 0; i < src.rows; i++)
    {
        // loop through each column
        for (int j = 0; j < src.cols; j++)
        {
            // make only green value the original green value and red/blue = 0
            dst.at<Vec3b>(i, j)[0] = 0;
            dst.at<Vec3b>(i, j)[2] = 0;
            dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
        }
    }
    return (0);
}

/**
 * This function takes the median of the values in a 3x3 matrix and applies that values to each pixel
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 */
int medianFilter(cv::Mat &src, cv::Mat &dst)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    Mat qtr; // image quarter of the original size
    resize(src, qtr, Size(src.cols / 4, src.rows / 4)); // initialize qtr image
    // create 8 bit unsigned char 3 channel destination matrix
    dst.create(qtr.size(), CV_8UC3);

    // loop through each pixel by row
    for (int i = 1; i < qtr.rows - 1; i++)
    {
        // get pointers for each row
        Vec3b *ptrup = qtr.ptr<Vec3b>(i - 1);
        Vec3b *ptrmd = qtr.ptr<Vec3b>(i);
        Vec3b *ptrdn = qtr.ptr<Vec3b>(i + 1);
        Vec3b *dptr = dst.ptr<Vec3b>(i);
        // loop through each column
        for (int j = 1; j < qtr.cols - 1; j++)
        {
            // loop through each channel
            for (int k = 0; k < qtr.channels(); k++)
            {
                // get all values in the 3x3 filter
                uchar values[9] = {ptrup[j - 1][k], ptrup[j][k], ptrup[j + 1][k],
                                   ptrmd[j - 1][k], ptrmd[j][k], ptrmd[j + 1][k],
                                   ptrdn[j - 1][k], ptrdn[j][k], ptrmd[j + 1][k]};

                sort(values, values + 9); // sort the values

                dptr[j][k] = values[4]; // assign value to destination pointer
            }
        }
    }
    // resize back to original size
    resize(dst, dst, cv::Size(src.cols, src.rows));
    return (0);
}

/**
 * This function covers detected faces
 * All values will be 255,255,255 to cover the faces
 * cv::Mat src - an unsigned short image taken as a source
 * cv::Mat dst - an unsigned short image to write the filter effect on
 * std::vector<cv::Rect> faces - vector of faces detected
 */
int faceCoverFilter(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces)
{
    // check if src is empty
    if (src.empty())
    {
        printf("No valid src file\n");
        return -1;
    }

    // create 8 bit unsigned char 3 channel destination matrix
    dst.create(src.size(), CV_8UC3);

    // loop through each row
    for (int i = 0; i < src.rows; ++i)
    {
        // loop through each column
        for (int j = 0; j < src.cols; ++j)
        {
            // check if pixel is inside detected face
            bool insideFace = false;

            // loop through each face
            for (const Rect &face : faces)
            {
                if (i >= face.y && i < face.y + face.height && j >= face.x && j < face.x + face.width) // checks if current pixel is within face
                {
                    insideFace = true;
                    break;
                }
            }

            // if pixel is inside the face, then set pixel value to 255,255,255
            if (insideFace)
            {

                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255); // Set to white
            }
            else
            {
                dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i, j); // Copy the original pixel value
            }
        }
    }

    return 0;
}
