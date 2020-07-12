#ifndef CONTOUR_HPP
#define CONTOUR_HPP

#include <opencv2/core.hpp>

#include <vector>

class Contour
{
    std::vector<cv::Point> _contourPoints;
    cv::Moments _m;
    bool _isConvex;

public:

    // create the Contour object given contour points
    Contour(std::vector<cv::Point> contourPoints);

    const bool& convex = _isConvex;

    // middle point of the bounding box
    cv::Point center() const;
    // mass center of contour area
    cv::Point weightedCenter() const;
    // pixel area
    double area() const;
    // bounding box
    cv::Rect box() const;
    // return contour points
    std::vector<cv::Point> contourPoints() const;
    // mask of contour with size of the bounding box
    cv::Mat getMask() const;
    // check whether point is inside the polygon
    bool contains(cv::Point2f pt) const;
};

#endif  // CONTOUR_HPP
