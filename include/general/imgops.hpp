#ifndef IMGOPS_HPP
#define IMGOPS_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <vector>

#include <general/Contour.hpp>

namespace imgops {

// resize with constant ratio
inline cv::Mat resize(const cv::Mat &img, int width) {
  cv::Mat resized;
  cv::Size dsize(width, img.rows * width / img.cols);
  cv::resize(img, resized, dsize);
  return resized;
}

// convert color image to grayscale
inline cv::Mat bgr2gray(const cv::Mat &img) {
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

// convert from grayscale to color image
inline cv::Mat gray2bgr(const cv::Mat &gray) {
  cv::Mat img;
  cv::cvtColor(gray, img, cv::COLOR_GRAY2BGR);
  return img;
}

// convert to HSV color space
inline cv::Mat bgr2hsv(const cv::Mat &img) {
  cv::Mat hsv;
  cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
  return hsv;
}

// convert from HSV color space
inline cv::Mat hsv2bgr(const cv::Mat &hsv) {
  cv::Mat img;
  cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);
  return img;
}

// obtain transparent image
// tr is between 0-1, 0 gives m1
inline cv::Mat blend(const cv::Mat &m1, const cv::Mat &m2, double tr = 0.5) {
  return m1 * (1 - tr) + m2 * tr;
}

// ROI without the black borders
cv::Rect cropBorder(const cv::Mat &img);

// add black borders to cover the ROI
// return the corresponding point of the old origin
cv::Point addBorder(cv::Mat &img, cv::Rect rect);

// falsecolor vector
std::vector<cv::Vec3b> colormap(cv::ColormapTypes type, bool shuffle = false);

// available optical flow types from *get_optflow*
std::vector<std::string> get_optflow_types();

// get the optical flow class at given type
// use *get_optflow_types* to see available types
// use_rgb: whether *calc* method of the returned class uses color or grayscale
// image
cv::Ptr<cv::DenseOpticalFlow> get_optflow(
    std::string optflowType,
    bool &use_rgb = const_cast<bool &>(static_cast<const bool &>(false)));

// draw optical flow vector field
void drawOptFlowMap(const cv::Mat &flow, cv::Mat &flowmap, int step = 16,
                    cv::Scalar color = CV_RGB(255, 0, 0));

// represent optical flow magnitude and angle in HSV color space
cv::Mat reprOptFlow(const cv::Mat &flow);

// available tracker types from *get_tracker*
std::vector<std::string> get_tracker_types();

// get the tracker class at given type
// use *get_tracker_types* to see available types
cv::Ptr<cv::Tracker> get_tracker(std::string trackerType);

} // namespace imgops

class ConnectImages {
  // ROI for train image
  cv::Rect _prevRect;
  // ROI for query image
  cv::Rect _nextRect;

public:
  // create ConnectImages class providing ROIs for train and query images
  ConnectImages(cv::Rect prevRect, cv::Rect nextRect);

  // create default ConnectImages class providing sizes of images
  ConnectImages(cv::Size prevSize, cv::Size nextSize);

  const cv::Rect &prevRect = _prevRect;
  const cv::Rect &nextRect = _nextRect;

  // get the principal Mat from images to compare
  cv::Mat compareImages(const cv::Mat &prevImg, const cv::Mat &nextImg) const;

  // draw lines between matching keypoints
  void connectCenters(cv::Mat &compImg, const std::vector<int> &trainIDs,
                      const std::vector<int> &queryIDs,
                      const std::vector<cv::KeyPoint> &trainKps,
                      const std::vector<cv::KeyPoint> &queryKps,
                      std::vector<bool> status = std::vector<bool>(),
                      bool markCenters = true) const;

  // connect matching contour centers
  void connectCenters(cv::Mat &compImg,
                      const std::vector<Contour> &prevContours,
                      const std::vector<Contour> &nextContours,
                      const std::vector<int> &prevIDs,
                      const std::vector<int> &nextIDs,
                      const std::vector<cv::Vec3b> &nextColors,
                      bool invertedColor = false) const;

  // draw bounding boxes around matching contours
  // leave previous or next parameters empty if you do not want to render on one
  // of the subfigures
  void drawBoxes(cv::Mat &compImg, const std::vector<Contour> &prevContours,
                 const std::vector<Contour> &nextContours,
                 const std::vector<int> &prevIDs,
                 const std::vector<int> &nextIDs,
                 const std::vector<cv::Vec3b> &prevColors,
                 const std::vector<cv::Vec3b> &nextColors,
                 bool invertedColor = false) const;

  // draw matching contours
  // leave previous or next parameters empty if you do not want to render on one
  // of the subfigures
  void drawContours(cv::Mat &compImg, const std::vector<Contour> &prevContours,
                    const std::vector<Contour> &nextContours,
                    const std::vector<int> &prevIDs,
                    const std::vector<int> &nextIDs,
                    const std::vector<cv::Vec3b> &prevColors,
                    const std::vector<cv::Vec3b> &nextColors,
                    int thickness = cv::FILLED,
                    bool invertedColor = false) const;

  // put numbers given points
  // leave previous or next parameters empty if you do not want to render on one
  // of the subfigures provide color only if you want to put all numbers in that
  // color in that case, invertedColor will be ignored
  void numberPoints(cv::Mat &compImg, const std::vector<Contour> &prevContours,
                    const std::vector<Contour> &nextContours,
                    const std::vector<int> &prevIDs,
                    const std::vector<int> &nextIDs,
                    const std::vector<cv::Vec3b> &prevColors,
                    const std::vector<cv::Vec3b> &nextColors,
                    cv::Scalar color = cv::Scalar(0, 0, 0, 255),
                    bool invertedColor = false) const;
};

#endif // IMGOPS_HPP
