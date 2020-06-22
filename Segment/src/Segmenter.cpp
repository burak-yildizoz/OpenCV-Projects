#include <opencv2/imgproc.hpp>

#include <algorithm>

#include "Segmenter.hpp"
#include "egbis.h"	// Efficient Graph-Based Image Segmentation algorithm
#include "general.hpp"

Segmenter::Segmenter(const cv::Mat& img, float sigma, float k, int min_size) :
    _labelImage(img.size(), CV_8U), _sigma(sigma), _k(k), _min_size(min_size)
{
    // obtain segmentation results
    image<rgb> *nativeImage = convertMatToNativeImage(img);
    universe *u = segmentation(nativeImage, sigma, k, min_size, &_num_ccs);
    delete nativeImage;

    // create label image
    _labelvec.reserve(_num_ccs);
    for (int x = 0; x < _labelImage.cols; x++)
    {
        for (int y = 0; y < _labelImage.rows; y++)
        {
            // this number is unique for each label, but not consequtive
            int label = u->find(y * _labelImage.cols + x);
            if (std::find(_labelvec.begin(), _labelvec.end(), label) == _labelvec.end())
                // the label is new
                _labelvec.push_back(label);
            uchar id = (uchar)std::distance(_labelvec.begin(), std::find(_labelvec.begin(), _labelvec.end(), label));
            CHECK(id < _labelvec.size());	// make sure id is found
            _labelImage.at<uchar>(y, x) = id;
        }
    }
    delete u;
}

cv::Mat Segmenter::getLabelImage() const
{
    return _labelImage.clone();
}

std::vector<Contour> Segmenter::contoursOfLabels() const
{
    // find contour points of each label
    std::vector<Contour> labelContours;
    labelContours.reserve(_num_ccs);
    for (int i = 0; i < _num_ccs; i++)
    {
        // find contours of segment
        std::vector< std::vector<cv::Point> > contours;
        cv::findContours((cv::Mat)(_labelImage == i), contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        // get the one with maximum area
        std::vector<cv::Point> contour = contours.front();
        for (size_t j = 1; j < contours.size(); j++)
            if (cv::contourArea(contours[j]) > cv::contourArea(contour))
                contour = contours[j];
        labelContours.push_back(contour);
    }
    return labelContours;
}

int Segmenter::belongs(cv::Point pt) const
{
    CHECK(cv::Rect(cv::Point(0, 0), _labelImage.size()).contains(pt));
    return _labelImage.at<uchar>(pt);
}
