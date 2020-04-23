#ifndef CALIBRATE_HPP
#define CALIBRATE_HPP

// OpenCV includes
#include <opencv2/videoio.hpp>
// standard library
#include <vector>
#include <fstream>
#include <string>
#include <new>

class Calibrate
{
    int _numBoards;
    int _numCornersHor;
    int _numCornersVer;
    std::string _object_filename;
    std::string _image_filename;

protected:
    // helper function for *save_chessboard*
    template <class T>
    static void write_to_file(std::ofstream &ofs, const std::vector<std::vector<T>> &obj);

public:
    const int &numBoards = _numBoards; // number of chessboard images to be calibrated
    // if the chessboard is 8x8, then both number of corners should be 7
    const int &numCornersHor = _numCornersHor;             // number of corners along width
    const int &numCornersVer = _numCornersVer;             // number of corners along height
    const std::string &object_filename = _object_filename; // external filename for object points
    const std::string &image_filename = _image_filename;   // external filename for image points

    /** @brief Class for camera calibration
	@param numBoards number of chessboard images to be calibrated
	@param numCornersHor number of corners along width
	@param numCornersVer  number of corners along heigth
	@param object_filename The name of the external file that object_points will be saved & loaded
	@param image_filename The name of the external file that image_points will be saved & loaded
	@note The parameters specified here cannot be altered later
	*/
    explicit Calibrate(int numBoards, int numCornersHor = 9, int numCornersVer = 6,
                       const std::string &object_filename = "object_points.txt",
                       const std::string &image_filename = "image_points.txt");

    /** @brief Opens the camera and saves the chessboard properties to external files
	Press space key to save the chessboard properties
	Press S to save also the image
	The function returns the number of objects that are successfully saved
	@param capture VideoCapture object to get the image stream. If specified, it needs to be open. The default is camera #0
	@param delay Wait time between each frame
	@param image_name Name of the image to be saved. Default saves image1.jpg, image2.jpg, ...
	@param ext Extension of the image. The image is saved as image_name + #image + ext
	*/
    int save_chessboard(cv::VideoCapture *capture = new cv::VideoCapture(0), int delay = 1,
                        std::string image_name = "image", std::string ext = ".jpg") const;

    /** @brief This is an overloaded function, provided for convenience. It differs from the above function only in what argument(s) it accepts.
	@param chessboard_images Sequence that contains chessboard images
	@param display If true, the detected chessboard will be shown for each frame
	*/
    int save_chessboard(const std::vector<cv::Mat> &chessboard_images, bool display = false, int delay = 0) const;

    /** @brief Load the parameters from the external files
	The function returns true if the parameters loaded successfuly
	@param object_points The physical position of the corners (in 3D space)
	@param image_points The location of the corners on in the image (in 2 dimensions)
	*/
    bool load_chessboard(std::vector<std::vector<cv::Point3f>> &object_points,
                         std::vector<std::vector<cv::Point2f>> &image_points) const;

    /** @brief Uses cv::calibrateCamera, but with more default arguments
	@param object_points The physical position of the corners (in 3D space)
	@param image_points The location of the corners on in the image (in 2 dimensions)
	@param imageSize Size of the image
	@param cameraMatrix 3x3 matrix that represents intrinsic parameters. An empty matrix is converted to have fx=fy=1
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

#endif // CALIBRATE_HPP