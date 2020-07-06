#include <opencv2/imgproc.hpp>

#include "imgops.hpp"
#include "general.hpp"

namespace imgops
{
    cv::Mat resize(const cv::Mat &img, int width)
    {
        cv::Mat resized;
        cv::Size dsize(width, img.rows * ((double)width / img.cols));
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
}

ConnectImages::ConnectImages(cv::Rect prevRect, cv::Rect nextRect) :
_prevRect(prevRect), _nextRect(nextRect) {}

ConnectImages::ConnectImages(cv::Size prevSize, cv::Size nextSize) :
_prevRect(cv::Rect(0, nextSize.height, prevSize.width, prevSize.height)),
_nextRect(cv::Rect(0, 0, nextSize.width, nextSize.height)) {}

cv::Mat ConnectImages::compareImages(const cv::Mat& prevImg, const cv::Mat& nextImg)
{
    cv::Mat compImg = cv::Mat::zeros(MAX(_nextRect.height + _nextRect.y, _prevRect.height + _prevRect.y),
        MAX(_nextRect.width + _nextRect.x, _prevRect.width + _prevRect.x), CV_8UC3);
    prevImg.copyTo(compImg(_prevRect));
    nextImg.copyTo(compImg(_nextRect));
    return compImg;
}

void ConnectImages::connectCenters(cv::Mat& compImg, const std::vector<int>& trainIDs, const std::vector<int>& queryIDs,
    const std::vector<cv::KeyPoint>& trainKps, const std::vector<cv::KeyPoint>& queryKps,
    std::vector<bool> status, bool markCenters)
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
        cv::line(compImg, prevCenter + _prevRect.tl(), nextCenter + _nextRect.tl(), CV_RGB(0, 255, 0), 1, CV_AA);
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
            cv::circle(compImg, prevCenter + _prevRect.tl(), 2, CV_RGB(255, 0, 0), CV_FILLED, CV_AA);
            cv::circle(compImg, nextCenter + _nextRect.tl(), 2, CV_RGB(0, 0, 255), CV_FILLED, CV_AA);
        }
    }
}
