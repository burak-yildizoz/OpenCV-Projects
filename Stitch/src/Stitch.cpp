#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "Stitch.hpp"
#include "imgops.hpp"
#include "general.hpp"

Feature::Feature(const cv::Mat &gray)
{
    CHECK(gray.dims == 2);
    Stitcher::descriptor->detectAndCompute(gray, cv::noArray(), kps, desc);
}

std::vector<cv::Point2f> Feature::getKeypoints() const
{
    std::vector<cv::Point2f> res(kps.size());
    for (size_t i = 0; i < kps.size(); i++)
        res[i] = kps[i].pt;
    return res;
}

std::pair<cv::Mat, cv::Point> Stitcher::_calculatePano()
{
    const cv::Mat &prevPano = _lastPano.first;
    const cv::Point &prevOrig = _lastPano.second;
    const cv::Mat &img = _imgs.back();
    // correct the homography matrix
    cv::Mat corrH;
    const std::vector<int> &trainIDs = _matches.back().first;
    const std::vector<int> &queryIDs = _matches.back().second;
    std::vector<cv::Point2f> trainKeypoints = _features[_features.size() - 2].getKeypoints();
    const std::vector<cv::KeyPoint> &queryKps = _features.back().kps;
    if (trainIDs.size() > 4)
    {
        // construct the two sets of points
        std::vector<cv::Point> srcPoints, dstPoints;
        for (size_t i = 0; i < trainIDs.size(); i++)
        {
            srcPoints.push_back((cv::Point)trainKeypoints[trainIDs[i]] + prevOrig);
            dstPoints.push_back(queryKps[queryIDs[i]].pt);
        }
        // compute the homography between the two sets of points
        double reprojThresh = 4.0;
        corrH = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, reprojThresh);
    }
    // find corresponding points after homography
    cv::Point orig = warpRect(prevPano.size(), corrH).tl(); // warped top-left point of last pano
    cv::Mat pano = warpImage(prevPano, corrH, orig);        // desired pano
    cv::Rect nextRect(-orig, img.size());                   // ROI of last image
    cv::Point shift = imgops::addBorder(pano, nextRect);    // the amount of shift at top-left corner
    img.copyTo(pano(cv::Rect(shift - orig, img.size())));
    // crop the black borders from top-left
    cv::Rect bb = imgops::cropBorder(pano);
    pano = pano(bb);
    cv::Point newOrig = shift - orig - bb.tl();
    return std::make_pair(pano, newOrig);
}

std::pair<std::vector<int>, std::vector<int>> Stitcher::findMatch(const cv::Mat &prevDesc, const cv::Mat &nextDesc)
{
    // compute the raw matches
    std::vector<std::vector<cv::DMatch>> rawMatch;
    matcher->knnMatch(nextDesc, prevDesc, rawMatch, 2);
    // initialize the list of actual matches
    std::vector<int> trainIDs, queryIDs;
    for (std::vector<cv::DMatch> &m : rawMatch)
    {
        // ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        double ratio = 0.75;
        if ((m.size() == 2) && (m[0].distance < m[1].distance * ratio))
        {
            trainIDs.push_back(m[0].trainIdx);
            queryIDs.push_back(m[0].queryIdx);
        }
    }
    return std::make_pair(trainIDs, queryIDs);
}

std::pair<cv::Mat, std::vector<bool>> Stitcher::findHomography(
    const std::vector<int> &trainIDs, const std::vector<int> &queryIDs,
    const std::vector<cv::KeyPoint> &trainKps, const std::vector<cv::KeyPoint> &queryKps)
{
    // computing a homography requires at least 4 matches
    cv::Mat H;
    std::vector<bool> state;
    if (trainIDs.size() > 4)
    {
        // construct the two sets of points
        std::vector<cv::Point> srcPoints, dstPoints;
        for (size_t i = 0; i < trainIDs.size(); i++)
        {
            srcPoints.push_back(trainKps[trainIDs[i]].pt);
            dstPoints.push_back(queryKps[queryIDs[i]].pt);
        }
        // compute the homography between the two sets of points
        double reprojThresh = 4.0;
        cv::Mat mask;
        H = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, reprojThresh, mask);
        // find successfully matched keypoints
        state.resize(mask.rows);
        for (int i = 0; i < mask.rows; i++)
            state[i] = mask.at<bool>(i, 0);
    }
    return std::make_pair(H, state);
}

cv::Rect Stitcher::warpRect(cv::Size sz, const cv::Mat &H)
{
    std::vector<cv::Point2f> corners(4);
    corners[0] = cv::Point2f(0, 0);                // top left
    corners[1] = cv::Point2f(sz.width, 0);         // top right
    corners[2] = cv::Point2f(0, sz.height);        // bottom left
    corners[3] = cv::Point2f(sz.width, sz.height); // bottom right
    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(corners, warpedCorners, H);
    return cv::boundingRect(warpedCorners);
}

cv::Mat Stitcher::warpImage(const cv::Mat &img, const cv::Mat &H, cv::Point orig)
{
    // calculate the ROI of the result
    cv::Rect roi = warpRect(img.size(), H);
    // specify the top-left corner as origin if not specified
    if (orig == cv::Point())
        orig = roi.tl();
    // translate the origin of the result from orig to (0, 0)
    cv::Mat T = (cv::Mat_<double>(3, 3) << 1, 0, -orig.x,
                 0, 1, -orig.y,
                 0, 0, 1);
    cv::Mat corrH = T * H;
    // warp the image
    cv::Mat warped;
    cv::Size dsize = warpRect(img.size(), corrH).size(); // output size
    warpPerspective(img, warped, corrH, dsize);
    return warped;
}

cv::Mat Stitcher::stitch(const cv::Mat &prevImg, const cv::Mat &nextImg, const cv::Mat &H)
{
    cv::Point orig = warpRect(prevImg.size(), H).tl();
    cv::Mat res = warpImage(prevImg, H, orig);
    cv::Rect nextRect(-orig, nextImg.size());
    cv::Point shift = imgops::addBorder(res, nextRect);
    nextImg.copyTo(res(cv::Rect(shift - orig, nextImg.size())));
    return res;
}

const cv::Ptr<cv::ORB> Stitcher::descriptor = cv::ORB::create();
const cv::Ptr<cv::FlannBasedMatcher> Stitcher::matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)));

Stitcher::Stitcher(const cv::Mat &img) : _imgs(std::vector<cv::Mat>(1, img.clone())),
                                         _cumulativeH(std::vector<cv::Mat>(1, cv::Mat::eye(3, 3, CV_64F)))
{
    // convert to grayscale
    cv::Mat gray = imgops::rgb2gray(img);
    // detect keypoints and extract local invariant descriptors from them
    Feature feature(gray);

    // save the results
    _lastPano = std::make_pair(_imgs.front(), cv::Point(0, 0));
    _features.push_back(feature);
}

void Stitcher::add(const cv::Mat &img)
{
    // convert to grayscale
    cv::Mat gray = imgops::rgb2gray(img);
    // detect keypoints and extract local invariant descriptors from them
    Feature feature(gray);

    // find the matching features
    std::pair<std::vector<int>, std::vector<int>> match = findMatch(_features.back().desc, feature.desc);
    const std::vector<int> &trainIDs = match.first;
    const std::vector<int> &queryIDs = match.second;

    // compute the homography between the previous image
    std::pair<cv::Mat, std::vector<bool>> homography = findHomography(
        trainIDs, queryIDs, _features.back().kps, feature.kps);
    const cv::Mat &H = homography.first;
    const std::vector<bool> &state = homography.second;

    // compute the homography between all previous images with img
    for (cv::Mat &cH : _cumulativeH)
        cH = cH * H;

    // save the results
    _imgs.push_back(img.clone());
    _features.push_back(feature);
    _matches.push_back(match);
    _status.push_back(state);
    _homographies.push_back(H);
    _cumulativeH.push_back(cv::Mat::eye(3, 3, CV_64F));
    _lastPano = _calculatePano();
}

cv::Mat Stitcher::pano() const
{
    /* This method warps each image to the pano seperately
	// find the boundaries of the result
	int xmin = INT_MAX, xmax = INT_MIN, ymin = INT_MAX, ymax = INT_MIN;
	for (size_t i = 0; i < _imgs.size(); i++)
	{
		cv::Size sz = _imgs[i].size();
		const cv::Mat &cH = _cumulativeH[i];
		cv::Rect roi = warpRect(sz, cH);
		xmin = MIN(xmin, roi.x);
		xmax = MAX(xmax, roi.br().x);
		ymin = MIN(ymin, roi.y);
		ymax = MAX(ymax, roi.br().y);
	}
	cv::Point anchor(xmin, ymin);
	cv::Size finalSize(xmax - xmin, ymax - ymin);

	// warp each image to the pano
	cv::Mat result(finalSize, CV_8UC3, cv::Scalar(0, 0, 0));
	for (size_t i = 0; i < _imgs.size(); i++)
	{
		const cv::Mat &img = _imgs[i];
		const cv::Mat &cH = _cumulativeH[i];
		cv::Point orig = warpRect(img.size(), cH).tl();
		cv::Mat warped = warpImage(img, cH, orig);
		cv::Mat mask = imgops::rgb2gray(warped);
		warped.copyTo(result(cv::Rect(orig - anchor, warped.size())), mask);
	}

	return result;
	*/
    return _lastPano.first.clone();
}

cv::Mat Stitcher::newestStitch() const
{
    CHECK(_imgs.size() > 1);
    const cv::Mat &prevImg = _imgs[_imgs.size() - 2];
    const cv::Mat &nextImg = _imgs.back();
    const cv::Mat &H = _homographies.back();
    return stitch(prevImg, nextImg, H);
}

cv::Mat Stitcher::drawMatches() const
{
    CHECK(_imgs.size() > 1);
    const cv::Mat &prevImg = _imgs[_imgs.size() - 2];
    const cv::Mat &nextImg = _imgs[_imgs.size() - 1];

    // create default ConnectImages class
    ConnectImages ci(prevImg.size(), nextImg.size());
    cv::Mat vis = ci.compareImages(prevImg, nextImg);

    // connect matching keypoints
    const std::vector<int> &trainIDs = _matches.back().first;
    const std::vector<int> &queryIDs = _matches.back().second;
    ci.connectCenters(vis, trainIDs, queryIDs, _features[_features.size() - 2].kps,
                      _features.back().kps, _status.back());

    return vis;
}

cv::Mat Stitcher::prevImg() const
{
    CHECK(_imgs.size() > 1);
    return _imgs[_imgs.size() - 2].clone();
}
