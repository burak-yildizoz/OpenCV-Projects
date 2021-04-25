#include <numeric>
#include <string>

#include "imgops.hpp"
#include "general.hpp"

namespace imgops
{
    cv::Mat resize(const cv::Mat &img, int width)
    {
        cv::Mat resized;
        cv::Size dsize(width, img.rows * width / img.cols);
        cv::resize(img, resized, dsize);
        return resized;
    }

    cv::Mat rgb2gray(const cv::Mat &img)
    {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    cv::Rect cropBorder(const cv::Mat &img)
    {
        CHECK(img.type() == CV_8UC3);

        // vector with all non-black point positions
        std::vector<cv::Point> nonBlackList;
        nonBlackList.reserve(img.total());

        // add all non-black points to the vector
        for (int j = 0; j < img.rows; j++)
            for (int i = 0; i < img.cols; i++)
                // if not black: add to the list
                if (img.at<cv::Vec3b>(j, i) != cv::Vec3b(0, 0, 0))
                    nonBlackList.push_back(cv::Point(i, j));

        // create bounding rect around those points
        return cv::boundingRect(nonBlackList);
    }

    cv::Point addBorder(cv::Mat& img, cv::Rect rect)
    {
        int top = 0, bottom = 0, left = 0, right = 0;
        cv::Point tl = rect.tl();
        cv::Point br = rect.br();
        if (tl.y < 0)
            top = -tl.y;
        if (br.y > img.rows)
            bottom = br.y - img.rows;
        if (tl.x < 0)
            left = -tl.x;
        if (br.x > img.cols)
            right = br.x - img.cols;
        cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT);
        return cv::Point(left, top);
    }

    cv::Mat blend(const cv::Mat& m1, const cv::Mat& m2, double tr)
    {
        return m1*(1 - tr) + m2*tr;
    }

    std::vector<cv::Vec3b> colormap(cv::ColormapTypes type, bool shuffle)
    {
        uchar data[256];
        std::iota(data, data + 256, 0);
        cv::Mat map(256, 1, CV_8U, data);
        cv::Mat falsecolor;
        cv::applyColorMap(map, falsecolor, type);
        if (shuffle)
            cv::randShuffle(falsecolor);
        std::vector<cv::Vec3b> res(falsecolor.begin<cv::Vec3b>(), falsecolor.end<cv::Vec3b>());
        return res;
    }
}

ConnectImages::ConnectImages(cv::Rect prevRect, cv::Rect nextRect) :
    _prevRect(prevRect), _nextRect(nextRect) {}

ConnectImages::ConnectImages(cv::Size prevSize, cv::Size nextSize) :
    _prevRect(cv::Rect(0, nextSize.height, prevSize.width, prevSize.height)),
    _nextRect(cv::Rect(0, 0, nextSize.width, nextSize.height)) {}

cv::Mat ConnectImages::compareImages(const cv::Mat& prevImg, const cv::Mat& nextImg) const
{
    cv::Mat compImg = cv::Mat::zeros(MAX(_nextRect.height + _nextRect.y, _prevRect.height + _prevRect.y),
        MAX(_nextRect.width + _nextRect.x, _prevRect.width + _prevRect.x), CV_8UC3);
    prevImg.copyTo(compImg(_prevRect));
    nextImg.copyTo(compImg(_nextRect));
    return compImg;
}

void ConnectImages::connectCenters(cv::Mat& compImg,
    const std::vector<int>& trainIDs, const std::vector<int>& queryIDs,
    const std::vector<cv::KeyPoint>& trainKps, const std::vector<cv::KeyPoint>& queryKps,
    std::vector<bool> status,
    bool markCenters) const
{
    CHECK(trainIDs.size() == queryIDs.size());
    size_t sz = trainIDs.size();
    if (status == std::vector<bool>())
        status = std::vector<bool>(sz, true);
    CHECK(status.size() == sz);
    for (size_t i = 0; i < sz; i++)
    {
        if (!status[i])
            continue;
        const int& trainID = trainIDs[i];
        const int& queryID = queryIDs[i];
        cv::Point prevCenter = trainKps[trainID].pt;
        cv::Point nextCenter = queryKps[queryID].pt;
        cv::line(compImg, prevCenter + _prevRect.tl(), nextCenter + _nextRect.tl(), CV_RGB(0, 255, 0), 1, cv::LINE_AA);
    }
    if (markCenters)
    {
        for (size_t i = 0; i < sz; i++)
        {
            if (!status[i])
                continue;
            const int& trainID = trainIDs[i];
            const int& queryID = queryIDs[i];
            cv::Point prevCenter = trainKps[trainID].pt;
            cv::Point nextCenter = queryKps[queryID].pt;
            cv::circle(compImg, prevCenter + _prevRect.tl(), 2, CV_RGB(255, 0, 0), cv::FILLED, cv::LINE_AA);
            cv::circle(compImg, nextCenter + _nextRect.tl(), 2, CV_RGB(0, 0, 255), cv::FILLED, cv::LINE_AA);
        }
    }
}

void ConnectImages::connectCenters(cv::Mat& compImg,
    const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
    const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
    const std::vector<cv::Vec3b>& nextColors,
    bool invertedColor) const
{
    CHECK(prevIDs.size() == nextIDs.size());
    CHECK(nextColors.size() == nextContours.size());
    size_t sz = prevIDs.size();
    for (size_t i = 0; i < sz; i++)
    {
        const Contour& prevContour = prevContours[prevIDs[i]];
        const Contour& nextContour = nextContours[nextIDs[i]];
        cv::Point prevCenter = prevContour.center();
        cv::Point nextCenter = nextContour.center();
        const cv::Vec3b& origColor = nextColors[nextIDs[i]];
        cv::Vec3b color = invertedColor ? (cv::Vec3b::all(255) - origColor) : origColor;
        cv::line(compImg, prevCenter + _prevRect.tl(), nextCenter + _nextRect.tl(), color, 1, cv::LINE_AA);
    }
}

void ConnectImages::drawBoxes(cv::Mat& compImg,
    const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
    const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
    const std::vector<cv::Vec3b>& prevColors, const std::vector<cv::Vec3b>& nextColors,
    bool invertedColor) const
{
    CHECK(prevContours.size() == prevColors.size());
    CHECK(nextContours.size() == nextColors.size());
    for (size_t i = 0; i < prevIDs.size(); i++)
    {
        int prevID = prevIDs[i];
        const cv::Vec3b& origColor = prevColors[prevID];
        cv::Vec3b color = invertedColor ? (cv::Vec3b::all(255) - origColor) : origColor;
        const Contour& prevContour = prevContours[prevID];
        cv::Rect contourRect = prevContour.box();
        cv::rectangle(compImg, cv::Rect(_prevRect.tl() + contourRect.tl(), contourRect.size()), color);
    }
    for (size_t i = 0; i < nextIDs.size(); i++)
    {
        int nextID = nextIDs[i];
        const cv::Vec3b& origColor = nextColors[nextID];
        cv::Vec3b color = invertedColor ? (cv::Vec3b::all(255) - origColor) : origColor;
        const Contour& nextContour = nextContours[nextID];
        cv::Rect contourRect = nextContour.box();
        cv::rectangle(compImg, cv::Rect(_nextRect.tl() + contourRect.tl(), contourRect.size()), color);
    }
}

void ConnectImages::drawContours(cv::Mat& compImg,
    const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
    const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
    const std::vector<cv::Vec3b>& prevColors, const std::vector<cv::Vec3b>& nextColors,
    int thickness, bool invertedColor) const
{
    CHECK(prevContours.size() == prevColors.size());
    CHECK(nextContours.size() == nextColors.size());
    for (size_t i = 0; i < prevIDs.size(); i++)
    {
        int prevID = prevIDs[i];
        const cv::Vec3b& origColor = prevColors[prevID];
        cv::Vec3b color = invertedColor ? (cv::Vec3b::all(255) - origColor) : origColor;
        std::vector<cv::Point> contour = prevContours[prevID].contourPoints();
        cv::Point tl = _prevRect.tl();
        for (size_t i = 0; i < contour.size(); i++)
            contour[i] += tl;
        cv::drawContours(compImg, std::vector< std::vector<cv::Point> >(1, contour), 0, color, thickness);
    }
    for (size_t i = 0; i < nextIDs.size(); i++)
    {
        int nextID = nextIDs[i];
        const cv::Vec3b& origColor = nextColors[nextID];
        cv::Vec3b color = invertedColor ? (cv::Vec3b::all(255) - origColor) : origColor;
        std::vector<cv::Point> contour = nextContours[nextID].contourPoints();
        cv::Point tl = _nextRect.tl();
        for (size_t i = 0; i < contour.size(); i++)
            contour[i] += tl;
        cv::drawContours(compImg, std::vector< std::vector<cv::Point> >(1, contour), 0, color, thickness);
    }
}

namespace
{
    cv::Vec3b scalar2vec3b(cv::Scalar s)
    {
        return cv::Vec3b(static_cast<int>(s[0]), static_cast<int>(s[1]), static_cast<int>(s[2]));
    }
}

void ConnectImages::numberPoints(cv::Mat& compImg,
    const std::vector<Contour>& prevContours, const std::vector<Contour>& nextContours,
    const std::vector<int>& prevIDs, const std::vector<int>& nextIDs,
    const std::vector<cv::Vec3b>& prevColors, const std::vector<cv::Vec3b>& nextColors,
    cv::Scalar color, bool invertedColor) const
{
    const bool color_provided = color != cv::Scalar(0, 0, 0, 255);
    if (!color_provided)
    {
        CHECK(prevContours.size() == prevColors.size());
        CHECK(nextContours.size() == nextColors.size());
    }
    for (size_t i = 0; i < prevIDs.size(); i++)
    {
        int prevID = prevIDs[i];
        cv::Point pt = prevContours[prevID].center();
        cv::Vec3b targetColor = scalar2vec3b(color);
        if (!color_provided)
            targetColor = invertedColor ? (cv::Vec3b::all(255) - prevColors[prevID]) : prevColors[prevID];
        std::string num = std::to_string(prevID);
        cv::putText(compImg, num, pt + _prevRect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.4, targetColor, 1, cv::LINE_AA);
    }
    for (size_t i = 0; i < nextIDs.size(); i++)
    {
        int nextID = nextIDs[i];
        cv::Point pt = nextContours[nextID].center();
        cv::Vec3b targetColor = scalar2vec3b(color);
        if (!color_provided)
            targetColor = invertedColor ? (cv::Vec3b::all(255) - nextColors[nextID]) : nextColors[nextID];
        std::string num = std::to_string(nextID);
        cv::putText(compImg, num, pt + _nextRect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.4, targetColor, 1, cv::LINE_AA);
    }
}
