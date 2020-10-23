#ifndef FISHEYE_HPP
#define FISHEYE_HPP

// base class
#include "Calibrate.hpp"

class Fisheye : public Calibrate
{

public:
    /** @brief Class for fisheye camera calibration
	@param numBoards number of chessboard images to be calibrated
	@param numCornersHor number of corners along width
	@param numCornersVer  number of corners along heigth
	@param object_filename The name of the external file that object_points will be saved & loaded
	@param image_filename The name of the external file that image_points will be saved & loaded
	@note The parameters specified here cannot be altered later
	*/
    explicit Fisheye(int numBoards, int numCornersHor = 9, int numCornersVer = 6,
                     const std::string &object_filename = "object_points.txt",
                     const std::string &image_filename = "image_points.txt");

    /** @brief Calibrate fisheye camera
	@param object_points The physical position of the corners (in 3D space)
	@param image_points The location of the corners on in the image (in 2 dimensions)
	@param imageSize Size of the image
	@param cameraMatrix 3x3 matrix that represents intrinsic parameters
	@param distCoeffs Distortion coefficients
	*/
    static void calibrate_camera(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                                 const std::vector<std::vector<cv::Point2f>> &imagePoints,
                                 cv::Size imageSize,
                                 cv::Mat &cameraMatrix,
                                 cv::Mat &distCoeffs);

    /** @brief Given camera calibration parameters, undistorts the image and displays the result
	@param cameraMatrix 3x3 matrix that represents intrinsic parameters
	@param distCoeffs Camera distortion coefficients
	@param img Image to be undistorted
	@param winname Name of the window that the result will be displayed. If it is empty, the window is held until user presses a key, and img becomes the resulting image. Else, waitKey needs to be called
	@note Do not specify *winname* to wait until user presses a key
	*/
    static void display_undistorted(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                    cv::Mat &img, std::string winname = "");

    /** @brief This is an overloaded function, provided for convenience. It differs from the above function only in what argument(s) it accepts.
	@param cap The image stream that will be undistorted. If specified, it must be open
	@param winname Name of the window that the result will be displayed.
	*/
    static void display_undistorted(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                    cv::VideoCapture *cap = new cv::VideoCapture(0),
                                    std::string winname = "undistorted");
};

#endif // FISHEYE_HPP