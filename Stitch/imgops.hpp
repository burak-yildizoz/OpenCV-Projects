#ifndef IMGOPS_HPP
#define IMGOPS_HPP

#include <opencv2/core.hpp>

#include <vector>

namespace imgops
{

    // resize with constant ratio
    cv::Mat resize(const cv::Mat &img, int width);

    // convert from RGB to grayscale
    cv::Mat rgb2gray(const cv::Mat &img);

    // ROI without the black borders
    cv::Rect cropBorder(const cv::Mat &img);

    // add black borders to cover the ROI
    // return the corresponding point of the old origin
    cv::Point addBorder(cv::Mat& img, cv::Rect rect);

}   // imgops

class ConnectImages
{
    // ROI for train image
    cv::Rect _prevRect;
    // ROI for query image
    cv::Rect _nextRect;

public:

    // create ConnectImages class providing ROIs for train and query images
    ConnectImages(cv::Rect prevRect, cv::Rect nextRect);

    // create default ConnectImages class providing sizes of images
    ConnectImages(cv::Size prevSize, cv::Size nextSize);

    const cv::Rect& prevRect = _prevRect;
    const cv::Rect& nextRect = _nextRect;

    // get the principal Mat from images to compare
    cv::Mat compareImages(const cv::Mat& prevImg, const cv::Mat& nextImg);

    // draw lines between matching keypoints
    void connectCenters(cv::Mat& compImg, const std::vector<int>& trainIDs, const std::vector<int>& queryIDs,
        const std::vector<cv::KeyPoint>& trainKps, const std::vector<cv::KeyPoint>& queryKps,
        std::vector<bool> status = std::vector<bool>(), bool markCenters = true);
};

#endif // IMGOPS_HPP