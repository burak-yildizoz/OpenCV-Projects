// OpenCV includes
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
// standard library
#include <vector>
#include <fstream>
#include <string>
// handy tools such as DEBUG, CHECK, etc.
#include "general.hpp"
// header file
#include "Fisheye.hpp"

Fisheye::Fisheye(int numBoards, int numCornersHor, int numCornersVer,
                 const std::string &object_filename, const std::string &image_filename)
    : Calibrate(numBoards, numCornersHor, numCornersVer, object_filename, image_filename) {}

void Fisheye::calibrate_camera(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                               const std::vector<std::vector<cv::Point2f>> &imagePoints,
                               cv::Size imageSize)
{
    if (distCoeffs.empty())
        distCoeffs = cv::Mat(4, 1, CV_32FC1);
    std::vector<cv::Mat> rvecs, tvecs;
    int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND + cv::fisheye::CALIB_FIX_SKEW;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1e-6);
    std::cout << "Calibrating!" << std::endl;
    double rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                                        rvecs, tvecs, flags, criteria);
    std::cout << "Calibration done! rms = " << rms << std::endl;
    DEBUG(cameraMatrix);
}

void Fisheye::display_undistorted(cv::Mat &img, std::string winname)
{
    CHECK(!img.empty());
    bool show_once = winname.empty();
    if (show_once)
        winname = "undistorted";
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat::eye(3, 3, CV_32FC1),
                                         cameraMatrix, img.size(), CV_16SC2, map1, map2);

    cv::Mat imageUndistorted;
    cv::remap(img, imageUndistorted, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    imshow(winname, imageUndistorted);
    if (show_once)
    {
        char ch = cv::waitKey(0);
        CHECK(ch != 27);
        img = imageUndistorted.clone();
    }
}

void Fisheye::display_undistorted(cv::VideoCapture &cap, std::string winname)
{
    // make sure image stream is open
    CHECK(cap.isOpened());
    cv::Mat img;
    cap >> img;
    CHECK(!img.empty());
    CHECK(!winname.empty());

    // for display purposes
    cv::namedWindow(winname);

    char ch = 0;
    while (ch != 27) // ESC is pressed
    {
        if (img.empty())
        {
            std::cout << "End of image stream!" << std::endl;
            break;
        }
        display_undistorted(img, winname);
        ch = cv::waitKey();
        cap >> img;
    }
    cv::destroyWindow(winname);
}
