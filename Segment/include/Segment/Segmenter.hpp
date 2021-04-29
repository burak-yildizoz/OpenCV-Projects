#ifndef SEGMENTER_HPP
#define SEGMENTER_HPP

#include <opencv2/core.hpp>

#include <vector>

#include <Stitch/Contour.hpp>

class Segmenter {
  int _num_ccs;
  const float _sigma;
  const float _k;
  const int _min_size;
  cv::Mat _labelImage;
  std::vector<int> _labelvec;

public:
  // Create the first EGBIS version with standard values
  Segmenter(const cv::Mat &img, float sigma = 0.5, float k = 500,
            int min_size = 200);

  const int &numSegments = _num_ccs;
  const float &sigma = _sigma;
  const float &k = _k;
  const int &min_size = _min_size;

  // label image from 0 to numSegments
  cv::Mat getLabelImage() const;

  // return the Contour object of each label
  std::vector<Contour> contoursOfLabels() const;

  // return the label id of point
  int belongs(cv::Point pt) const;
};

#endif // SEGMENTER_HPP