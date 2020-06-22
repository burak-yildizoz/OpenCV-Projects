#include <opencv2/imgproc.hpp>

#include "Contour.hpp"

Contour::Contour(std::vector<cv::Point> contourPoints) :
    _contourPoints(contourPoints),
    _m(cv::moments(contourPoints)),
    _isConvex(cv::isContourConvex(contourPoints)) {}

cv::Point Contour::center() const
{
    cv::Rect br = cv::boundingRect(_contourPoints);
    return cv::Point(br.x + br.width / 2, br.y + br.height / 2);
}

double Contour::area() const
{
    return cv::contourArea(_contourPoints);
}

cv::Rect Contour::box() const
{
    return cv::boundingRect(_contourPoints);
}

cv::Point Contour::weightedCenter() const
{
    return cv::Point(_m.m10 / _m.m00, _m.m01 / _m.m00);
}

std::vector<cv::Point> Contour::contourPoints() const
{
    return _contourPoints;
}

cv::Mat Contour::getMask() const
{
    cv::Rect bb = box();
    cv::Mat mask = cv::Mat::zeros(bb.size(), CV_8U);
    std::vector<cv::Point> contour = _contourPoints;
    cv::Point tl = bb.tl();
    for (cv::Point& pt : contour)
        pt -= tl;
    if (_isConvex)
        cv::fillConvexPoly(mask, contour, cv::Scalar::all(255));
    else
        cv::fillPoly(mask, std::vector< std::vector<cv::Point> >(1, contour), cv::Scalar::all(255));
    return mask;
}

bool Contour::contains(cv::Point2f pt) const
{
    double dist = cv::pointPolygonTest(_contourPoints, pt, false);
    return dist >= 0;
}
