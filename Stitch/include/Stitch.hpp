#ifndef STITCH_HPP
#define STITCH_HPP

#include <opencv2/core.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#else
#include "general.hpp"
STATIC_WARNING(false, "OpenCV xfeatures2d is not available. Using ORB features."
                      " There may occur incorrect results more than expected.");
#include <opencv2/features2d.hpp>
#endif
#include <functional>
#include <utility>
#include <vector>

struct Feature {
  // keypoints
  std::vector<cv::KeyPoint> kps;
  // descriptors
  cv::Mat desc;
  // obtain the features of an image
  Feature(const cv::Mat &gray);
  // default destructor
  ~Feature() {}
  // return the vector of keypoints
  std::vector<cv::Point2f> getKeypoints() const;
};

class Stitcher {
  // images to be stitched
  std::vector<cv::Mat> _imgs;
  // corresponding features of each image
  std::vector<Feature> _features;
  // actual matches between previous and next images
  // {trainIdx, queryIdx}
  // see findMatch
  // size: img_size - 1
  std::vector<std::pair<std::vector<int>, std::vector<int>>> _matches;
  // successful matching results in _matches
  // see findHomography
  // size: img_size - 1
  std::vector<std::vector<bool>> _status;
  // homography matrices from the previous image to the next one
  // see findHomography
  // size: img_size - 1
  std::vector<cv::Mat> _homographies;
  // final stitching result using the previous one
  // {pano, orig}
  // orig is the top-left point of the last inserted image
  // see warpPano
  std::pair<cv::Mat, cv::Point> _lastPano;

  // calculate the final stitching result and the origin of last image
  // {pano, orig}
  // see warpPano
  std::pair<cv::Mat, cv::Point> _calculatePano();

public:
  // detect and extract features from the image
#ifdef HAVE_OPENCV_XFEATURES2D
  static const cv::Ptr<cv::xfeatures2d::SIFT> descriptor;
#else
  static const cv::Ptr<cv::ORB> descriptor;
#endif
  // compute the raw matches
  static const cv::Ptr<cv::FlannBasedMatcher> matcher;

  // returns IDs of matched features given descriptors
  // {trainIds, queryIds}
  static std::pair<std::vector<int>, std::vector<int>>
  findMatch(const cv::Mat &prevDesc, const cv::Mat &nextDesc);
  // returns the 3x3 homography matrix given matching keypoints
  // and the success status of given matching IDs
  // {H, state}
  static std::pair<cv::Mat, std::vector<bool>>
  findHomography(const std::vector<int> &trainIDs,
                 const std::vector<int> &queryIDs,
                 const std::vector<cv::Point2f> &trainPts,
                 const std::vector<cv::Point2f> &queryPts);
  // find the ROI of an affine transform result
  // sz: input image size
  // H: homography matrix
  static cv::Rect warpRect(cv::Size sz, const cv::Mat &H);
  // apply affine transform given homography matrix and the top-left coordinate
  // orig orig is moved to (0, 0) to be covered in the result by applying
  // translation to homography matrix if orig is not specified, then the result
  // of warpRect is used
  static cv::Mat warpImage(const cv::Mat &img, const cv::Mat &H,
                           cv::Point orig = cv::Point());
  // stitch two images without losing the points that falls outside the first
  // quadrant
  static cv::Mat stitch(const cv::Mat &prevImg, const cv::Mat &nextImg,
                        const cv::Mat &H);
  // paste the previous pano onto the next one and return the corresponding
  // coordinate of the given point the given pixel coordinates must be the
  // origin of the same image
  static std::pair<cv::Mat, cv::Point>
  patchPano(const std::pair<cv::Mat, cv::Point> &prevPano,
            const std::pair<cv::Mat, cv::Point> &nextPano);
  // stitch the last pano onto the new image
  // the corrected homography matrix must be calculated previously and it must
  // directly warp prevPano onto img
  static std::pair<cv::Mat, cv::Point>
  warpPano(const cv::Mat &img, const cv::Mat &prevPano, const cv::Mat &corrH);
  // combine images given the local map and the function which gives the image
  // at a specific index Im(x, y) should return the image that
  // localMap.at<uchar>(y, x) corresponds to
  static cv::Mat combineImages(const cv::Mat &localMap,
                               std::function<cv::Mat(int, int)> Im);

  // construct the Stitcher object with the first image
  Stitcher(const cv::Mat &img);
  // construct a new Stitcher object with given previous pano
  // stitching is done from lastPano(lastRect) to img
  Stitcher(const cv::Mat &img, const cv::Mat &lastPano, cv::Rect lastRect);
  // construct a new Stitcher object by patching two Stitchers
  // see patchPano
  Stitcher(const Stitcher &prevStitcher, const Stitcher &nextStitcher);
  // default destructor
  ~Stitcher() {}

  // add a new image to get stitching results
  void add(const cv::Mat &img);
  // get the overall stitching result
  const cv::Mat pano() const;
  // get the last stitching result
  cv::Mat newestStitch() const;
  // show the last matched keypoints
  cv::Mat drawMatches() const;
  // the final stitching result with origin point of last image
  std::pair<cv::Mat, cv::Point> panoWithOrigin() const;
  // last image
  cv::Mat lastImg() const;
};

#endif // STITCH_HPP
