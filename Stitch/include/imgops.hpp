#ifndef IMGOPS_HPP
#define IMGOPS_HPP

#include <opencv2/imgproc.hpp>

#include <vector>

#include "Contour.hpp"

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

    // obtain transparent image
    // tr is between 0-1, 0 gives m1
    cv::Mat blend(const cv::Mat& m1, const cv::Mat& m2, double tr = 0.5);

    // falsecolor vector
    std::vector<cv::Vec3b> colormap(cv::ColormapTypes type, bool shuffle = false);

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
    cv::Mat compareImages(const cv::Mat& prevImg, const cv::Mat& nextImg) const;

    // draw lines between matching keypoints
    void connectCenters(cv::Mat& compImg,
        const std::vector<int>& trainIDs, const std::vector<int>& queryIDs,
        const std::vector<cv::KeyPoint>& trainKps, const std::vector<cv::KeyPoint>& queryKps,
        std::vector<bool> status = std::vector<bool>(),
        bool markCenters = true) const;

    // connect matching contour centers
    void connectCenters(cv::Mat& compImg,
        const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
        const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
        const std::vector<cv::Vec3b>& nextColors,
        bool invertedColor = false) const;

    // draw bounding boxes around matching contours
    // leave previous or next parameters empty if you do not want to render on one of the subfigures
    void drawBoxes(cv::Mat& compImg,
        const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
        const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
        const std::vector<cv::Vec3b>& prevColors, const std::vector<cv::Vec3b>& nextColors,
        bool invertedColor = false) const;

    // draw matching contours
    // leave previous or next parameters empty if you do not want to render on one of the subfigures
    void drawContours(cv::Mat& compImg,
        const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
        const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
        const std::vector<cv::Vec3b>& prevColors, const std::vector<cv::Vec3b>& nextColors,
        int thickness = cv::FILLED, bool invertedColor = false) const;

    // put numbers given points
    // leave previous or next parameters empty if you do not want to render on one of the subfigures
    // provide color only if you want to put all numbers in that color
    // in that case, invertedColor will be ignored
    void numberPoints(cv::Mat& compImg,
        const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
        const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
        const std::vector<cv::Vec3b>& prevColors, const std::vector<cv::Vec3b>& nextColors,
        cv::Scalar color = cv::Scalar(0, 0, 0, 255), bool invertedColor = false) const;
};

#endif // IMGOPS_HPP
