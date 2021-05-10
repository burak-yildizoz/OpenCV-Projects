#ifndef CALIBRATE_HPP
#define CALIBRATE_HPP

// OpenCV includes
#include <opencv2/videoio.hpp>
// standard library
#include <string>
#include <vector>

class Calibrate {
  int _numBoards; // number of chessboard images to be calibrated
  // if the chessboard is 8x8, then both number of corners should be 7
  int _numCornersHor;           // number of corners along width
  int _numCornersVer;           // number of corners along height
  std::string _object_filename; // external filename for object points
  std::string _image_filename;  // external filename for image points

protected:
  // helper function for *save_chessboard*
  template <class T>
  static void write_to_file(std::string filename,
                            const std::vector<std::vector<T>> &obj);
  cv::Mat cameraMatrix;
  cv::Mat distCoeffs;

public:
  /** @brief Class for camera calibration
      @param numBoards number of chessboard images to be calibrated
      @param numCornersHor number of corners along width
      @param numCornersVer  number of corners along heigth
      @param object_filename The name of the external file that object_points
     will be saved & loaded
      @param image_filename The name of the external file that image_points will
     be saved & loaded
      @note The parameters specified here cannot be altered later
      */
  Calibrate(int numBoards, int numCornersHor = 9, int numCornersVer = 6,
            std::string object_filename = "object_points.txt",
            std::string image_filename = "image_points.txt");

  /** @brief Saves the chessboard properties to external files from given image
     stream Press space key to save the chessboard properties Press S to save
     also the image The function returns the number of objects that are
     successfully saved
      @param cap VideoCapture object to get the image stream. It must be opened.
      @param delay Wait time between each frame
      @param image_name Name of the image to be saved. Default saves image1.jpg,
     image2.jpg, ...
      @param ext Extension of the image. The image is saved as image_name +
     #image + ext
      */
  int save_chessboard(cv::VideoCapture &cap, int delay = 1,
                      std::string image_name = "image",
                      std::string ext = ".jpg") const;

  /** @brief Load the parameters from the external files
      The function returns true if the parameters loaded successfuly
      @param object_points The physical position of the corners (in 3D space)
      @param image_points The location of the corners on in the image (in 2
     dimensions)
      */
  bool
  load_chessboard(std::vector<std::vector<cv::Point3f>> &object_points,
                  std::vector<std::vector<cv::Point2f>> &image_points) const;

  /** @brief Uses cv::calibrateCamera, but with more default arguments
      @param object_points The physical position of the corners (in 3D space)
      @param image_points The location of the corners on in the image (in 2
     dimensions)
      @param imageSize Size of the image
      @param cameraMatrix 3x3 matrix that represents intrinsic parameters. An
     empty matrix is converted to have fx=fy=1
      @param distCoeffs Distortion coefficients
      */
  virtual void
  calibrate_camera(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                   const std::vector<std::vector<cv::Point2f>> &imagePoints,
                   cv::Size imageSize);

  /** @brief Given camera calibration parameters, undistorts the image and
     displays the result
      @param cameraMatrix 3x3 matrix that represents intrinsic parameters
      @param distCoeffs Camera distortion coefficients
      @param img Image to be undistorted
      @param winname Name of the window that the result will be displayed. If it
     is empty, the window is held until user presses a key, and img becomes the
     resulting image. Else, waitKey needs to be called
      @note Do not specify *winname* to wait until user presses a key
      */
  virtual void display_undistorted(cv::Mat &img,
                                   std::string winname = "") const;

  /** @brief Given camera calibration parameters, undistorts the images and
     displays the result
      @param cap The image stream that will be undistorted. It must be opened.
      @param delay Wait time in ms between frames. 0 means wait until a
     keystroke.
      @param winname Name of the window that the result will be displayed.
      */
  void display_undistorted_all(
      cv::VideoCapture &cap,
      const std::vector<std::vector<cv::Point2f>> &imagePoints, int delay = 1,
      std::string winname = "undistorted") const;

  void displayImagePoints(cv::Mat &img,
                          const std::vector<cv::Point2f> &imagePoints) const;

  void calculateImagePoints(
      const std::vector<std::vector<cv::Point2f>> &imagePoints,
      const std::vector<std::vector<cv::Point3f>> &objectPoints,
      std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs) const;
};

#endif // CALIBRATE_HPP
