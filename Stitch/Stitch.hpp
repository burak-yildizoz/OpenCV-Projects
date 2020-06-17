#ifndef STITCH_HPP
#define STITCH_HPP

#include <opencv2/features2d.hpp>
#include <vector>
#include <utility>

struct Feature
{
	// keypoints
	std::vector<cv::KeyPoint> kps;
	// descriptors
	cv::Mat desc;
	// obtain the features of an image
	Feature(const cv::Mat& gray);
	// return the vector of keypoints
	std::vector<cv::Point2f> getKeypoints() const;
};

class Stitcher
{
	// images to be stitched
	std::vector<cv::Mat> _imgs;
	// corresponding features of each image
	std::vector<Feature> _features;
	// actual matches between previous and next images
	// {trainIdx, queryIdx}
	// size: img_size - 1
	std::vector<std::pair<std::vector<int>, std::vector<int>>> _matches;
	// successful matching results in _matches
	// size: img_size - 1
	std::vector<std::vector<bool>> _status;
	// homography matrices from the previous image to the next one
	// size: img_size - 1
	std::vector<cv::Mat> _homographies;
	// cumulative homography matrices to warp each image to the newest image
	// size: img_size
	// Note: _cumulativeH.back() = cv::Mat::eye(3, 3, CV_64F)
	std::vector<cv::Mat> _cumulativeH;
	// final stitching result using the previous one
	// {pano, orig}
	// orig is the top-left point of the last inserted image
	std::pair<cv::Mat, cv::Point> _lastPano;

	// calculate the final stitching result and the origin of last image
	// {pano, orig}
	std::pair<cv::Mat, cv::Point> _calculatePano();

public:

	// detect and extract features from the image
	static const cv::Ptr<cv::ORB> descriptor;
	// compute the raw matches
	static const cv::Ptr<cv::FlannBasedMatcher> matcher;

	// returns IDs of matched features given descriptors
	// {trainIds, queryIds}
	std::pair<std::vector<int>, std::vector<int>> findMatch(const cv::Mat& prevDesc, const cv::Mat& nextDesc);
	// returns the 3x3 homography matrix given matching keypoints
	// and the success status of given matching IDs
	// {H, state}
	std::pair<cv::Mat, std::vector<bool>> findHomography(
		const std::vector<int>& trainIDs, const std::vector<int>& queryIDs,
		const std::vector<cv::KeyPoint>& trainKps, const std::vector<cv::KeyPoint>& queryKps);
	// find the ROI of an affine transform result
	// sz: input image size
	// H: homography matrix
	static cv::Rect warpRect(cv::Size sz, const cv::Mat &H);
	// apply affine transform given the top-left coordinate
	// orig is moved to (0, 0) to be covered in the result
	// if orig is not specified, then the result of warpRect is used
	static cv::Mat warpImage(const cv::Mat &img, const cv::Mat &H, cv::Point orig = cv::Point());
	// stitch two images without losing points outside first quadrant
	static cv::Mat stitch(const cv::Mat& prevImg, const cv::Mat& nextImg, const cv::Mat& H);

	// construct the Stitcher object with the first image
	Stitcher(const cv::Mat &img);

	// add a new image
	void add(const cv::Mat &img);
	// get the final stitching result
	cv::Mat pano() const;
	// get the last stitching result
	cv::Mat newestStitch() const;
	// show the matched keypoints
	cv::Mat drawMatches() const;
	// the image before the last one
	cv::Mat prevImg() const;
};

#endif // STITCH_CPP
