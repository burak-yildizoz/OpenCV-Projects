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
    const std::vector<int> &trainIDs = _matches.back().first;
    const std::vector<int> &queryIDs = _matches.back().second;
    std::vector<cv::Point2f> trainPts = _features[_features.size() - 2].getKeypoints();
    std::vector<cv::Point2f> queryPts = _features.back().getKeypoints();
    for (cv::Point2f& pt : trainPts)
        pt += (cv::Point2f)prevOrig;
    cv::Mat corrH = findHomography(trainIDs, queryIDs, trainPts, queryPts).first;

    return warpPano(img, prevPano, corrH);
}

std::pair<std::vector<int>, std::vector<int>> Stitcher::findMatch(const cv::Mat &prevDesc, const cv::Mat &nextDesc)
{
    // compute the raw matches
    std::vector<std::vector<cv::DMatch>> rawMatch;
    if (prevDesc.rows > 2 && nextDesc.rows > 2)
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
    const std::vector<cv::Point2f> &trainPts, const std::vector<cv::Point2f> &queryPts)
{
    // computing a homography requires at least 4 matches
    cv::Mat H;
    std::vector<bool> state;
    if (trainIDs.size() > 4)
    {
        // construct the two sets of points
        std::vector<cv::Point2f> srcPoints, dstPoints;
        for (size_t i = 0; i < trainIDs.size(); i++)
        {
            srcPoints.push_back(trainPts[trainIDs[i]]);
            dstPoints.push_back(queryPts[queryIDs[i]]);
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
    CHECK(!H.empty());
    return std::make_pair(H, state);
}

cv::Rect Stitcher::warpRect(cv::Size sz, const cv::Mat &H)
{
    std::vector<cv::Point2f> corners(4);
    float w = static_cast<float>(sz.width);
    float h = static_cast<float>(sz.height);
    corners[0] = cv::Point2f(0, 0); // top left
    corners[1] = cv::Point2f(w, 0); // top right
    corners[2] = cv::Point2f(0, h); // bottom left
    corners[3] = cv::Point2f(w, h); // bottom right
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

std::pair<cv::Mat, cv::Point> Stitcher::patchPano(
    const std::pair<cv::Mat, cv::Point>& prevPano,
    const std::pair<cv::Mat, cv::Point>& nextPano)
{
    const cv::Mat& prevImg = prevPano.first;
    const cv::Mat& nextImg = nextPano.first;
    const cv::Point& prevOrig = prevPano.second;
    const cv::Point& nextOrig = nextPano.second;
    // bottom right points
    cv::Point prevbr = prevImg.size();
    cv::Point nextbr = nextImg.size();
    // distance from orig to br
    cv::Point nextdiag = nextbr - nextOrig;
    // possible pano corner coordinates
    std::vector<cv::Point> cornerpoints(4);
    cornerpoints[0] = cv::Point(0, 0);
    cornerpoints[1] = prevbr;
    cornerpoints[2] = prevOrig + nextdiag;
    cornerpoints[3] = prevOrig - nextOrig;
    cv::Rect bb = cv::boundingRect(cornerpoints);
    // patch prevImg to nextImg
    cv::Mat pano = prevImg.clone();
    cv::Point shift = imgops::addBorder(pano, bb);
    cv::Point orig = prevOrig + shift;
    nextImg.copyTo(pano(cv::Rect(orig - nextOrig, nextImg.size())), imgops::rgb2gray(nextImg));
    return std::make_pair(pano, orig);
}

std::pair<cv::Mat, cv::Point> Stitcher::warpPano(const cv::Mat& img, const cv::Mat& prevPano, const cv::Mat& corrH)
{
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

cv::Mat Stitcher::combineImages(const cv::Mat& localMap, std::function<cv::Mat(int, int)> Im)
{
    // find the centroid of the contour of the area in localMap
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(localMap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    Contour contour(contours.front());
    cv::Point center = contour.weightedCenter();

    // define some handy functions
    auto startPoint = [&localMap](int row) -> int
    {
        for (int i = 0; i < localMap.cols; i++)
            if (localMap.at<uchar>(row, i))
                return i;
        CHECK(false);
        return -1;
    };
    auto endPoint = [&localMap](int row) -> int
    {
        for (int i = localMap.cols - 1; i >= 0; i--)
            if (localMap.at<uchar>(row, i))
                return i;
        CHECK(false);
        return -1;
    };
    auto closestPoint = [&localMap, &center](int row) -> int
    {
        int idx = -1;
        for (int i = 0; i < localMap.cols; i++)
        {
            if (localMap.at<uchar>(row, i))
            {
                if (idx == -1)
                    idx = i;
                else if (abs(center.x - i) < abs(center.x - idx))
                    idx = i;
                else
                    return idx;
            }
        }
        CHECK(idx != -1);
        return idx;
    };

    // obtain the pano of the upper part
    int lastNode = closestPoint(0);
    Stitcher warpedRowPanoTop(Im(lastNode, 0));
    for (int i = 0; i < center.y; i++)
    {
        int node = closestPoint(i);
        int start = startPoint(i);
        int end = endPoint(i);
        Stitcher leftPano(Im(start, i));
        // stitch from start to node
        for (int j = start + 1; j <= node; j++)
        {
            leftPano.add(Im(j, i));
            // patch warpedRowPanoTop at the specific point
            if (j == lastNode)
                leftPano = Stitcher(leftPano, warpedRowPanoTop);
        }
        Stitcher rightPano(Im(end, i));
        // stitch from end to node
        for (int j = end - 1; j >= node; j--)
        {
            rightPano.add(Im(j, i));
            // patch warpedRowPanoTop at the specific point
            if (j == lastNode)
                rightPano = Stitcher(rightPano, warpedRowPanoTop);
        }
        // Note that even if we have patched warpedRowPanoTop to both leftPano and rightPano
        // when lastNode is node, it does not affect the result of patching them to rowPano
        // since they won't be warped after warpedRowPanoTop is patched
        Stitcher rowPano(leftPano, rightPano);
        std::pair<cv::Mat, cv::Point> pano = rowPano.panoWithOrigin();
        warpedRowPanoTop = Stitcher(Im(node, i + 1), pano.first, cv::Rect(pano.second, rowPano.lastImg().size()));
        lastNode = node;
    }

    // obtain the pano of the lower part
    lastNode = closestPoint(localMap.rows - 1);
    Stitcher warpedRowPanoBottom(Im(lastNode, localMap.rows - 1));
    for (int i = localMap.rows - 1; i > center.y; i--)
    {
        int node = closestPoint(i);
        int start = startPoint(i);
        int end = endPoint(i);
        Stitcher leftPano(Im(start, i));
        // stitch from start to node
        for (int j = start + 1; j <= node; j++)
        {
            leftPano.add(Im(j, i));
            // patch warpedRowPanoBottom at the specific point
            if (j == lastNode)
                leftPano = Stitcher(leftPano, warpedRowPanoBottom);
        }
        Stitcher rightPano(Im(end, i));
        // stitch from end to node
        for (int j = end - 1; j >= node; j--)
        {
            rightPano.add(Im(j, i));
            // patch warpedRowPanoBottom at the specific point
            if (j == lastNode)
                rightPano = Stitcher(rightPano, warpedRowPanoBottom);
        }
        // Note that even if we have patched warpedRowPanoBottom to both leftPano and rightPano
        // when lastNode is node, it does not affect the result of patching them to rowPano
        // since they won't be warped after warpedRowPanoBottom is patched
        Stitcher rowPano(leftPano, rightPano);
        std::pair<cv::Mat, cv::Point> pano = rowPano.panoWithOrigin();
        warpedRowPanoBottom = Stitcher(Im(node, i - 1), pano.first, cv::Rect(pano.second, rowPano.lastImg().size()));
        lastNode = node;
    }

    // obtain the pano of the row at center
    int node = center.x;
    int start = startPoint(center.y);
    int end = endPoint(center.y);
    Stitcher leftPano(Im(start, center.y));
    // stitch from start to node
    for (int j = start + 1; j <= node; j++)
        leftPano.add(Im(j, center.y));
    Stitcher rightPano(Im(end, center.y));
    // stitch from end to node
    for (int j = end - 1; j >= node; j--)
        rightPano.add(Im(j, center.y));
    Stitcher rowPano(leftPano, rightPano);

    // obtain the final pano by patching the three pano
    Stitcher finalPano(warpedRowPanoTop, warpedRowPanoBottom);
    finalPano = Stitcher(rowPano, finalPano);

    return finalPano.pano();
}

#ifdef HAVE_OPENCV_XFEATURES2D
    const cv::Ptr<cv::xfeatures2d::SIFT> Stitcher::descriptor = cv::xfeatures2d::SIFT::create();
    const cv::Ptr<cv::FlannBasedMatcher> Stitcher::matcher = cv::FlannBasedMatcher::create();
#else
    const cv::Ptr<cv::ORB> Stitcher::descriptor = cv::ORB::create();
    const cv::Ptr<cv::FlannBasedMatcher> Stitcher::matcher = cv::makePtr<cv::FlannBasedMatcher>(
        cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)));
#endif

Stitcher::Stitcher(const cv::Mat &img) :
    _imgs(std::vector<cv::Mat>(1, img.clone()))
{
    // convert to grayscale
    cv::Mat gray = imgops::rgb2gray(img);
    // detect keypoints and extract local invariant descriptors from them
    Feature feature(gray);

    // save the results
    _lastPano = std::make_pair(_imgs.front(), cv::Point(0, 0));
    _features.push_back(feature);
}

Stitcher::Stitcher(const cv::Mat& img, const cv::Mat& lastPano, cv::Rect lastRect)
{
    cv::Mat prevImg = lastPano(lastRect).clone();
    _imgs.resize(2);
    _imgs[0] = prevImg;
    _imgs[1] = img.clone();

    // find features
    Feature prevFeature(imgops::rgb2gray(prevImg));
    Feature feature(imgops::rgb2gray(img));
    _features.push_back(prevFeature);
    _features.push_back(feature);

    // find the matching features
    std::pair<std::vector<int>, std::vector<int>> match = findMatch(prevFeature.desc, feature.desc);
    const std::vector<int> &trainIDs = match.first;
    const std::vector<int> &queryIDs = match.second;
    _matches.push_back(match);

    // compute the homography between the previous image
    std::vector<cv::Point2f> trainPts = prevFeature.getKeypoints();
    std::vector<cv::Point2f> queryPts = feature.getKeypoints();
    std::pair<cv::Mat, std::vector<bool>> homography = findHomography(trainIDs, queryIDs, trainPts, queryPts);
    _homographies.push_back(homography.first);
    _status.push_back(homography.second);

    // calculate the last pano
    for (cv::Point2f& pt : trainPts)
        pt += (cv::Point2f)lastRect.tl();
    cv::Mat corrH = findHomography(trainIDs, queryIDs, trainPts, queryPts).first;
    _lastPano = warpPano(img, lastPano, corrH);
}

Stitcher::Stitcher(const Stitcher& prevStitcher, const Stitcher& nextStitcher)
{
    // check if they have the same lastImg
    cv::Mat img = prevStitcher.lastImg();
    auto equal = [](const cv::Mat& lhs, const cv::Mat& rhs) -> bool
    {
        if ((lhs.rows != rhs.rows) || (lhs.cols != rhs.cols))
            return false;
        cv::Scalar s = cv::sum(lhs - rhs);
        return (s[0] == 0) && (s[1] == 0) && (s[2] == 0);
    };
    CHECK(equal(img, nextStitcher.lastImg()));
    // patch the panos
    std::pair<cv::Mat, cv::Point> prevPano = prevStitcher.panoWithOrigin();
    std::pair<cv::Mat, cv::Point> nextPano = nextStitcher.panoWithOrigin();
    std::pair<cv::Mat, cv::Point> pano = Stitcher::patchPano(prevPano, nextPano);
    // save the results
    _imgs.push_back(img);
    _lastPano = pano;
    _features.push_back(Feature(imgops::rgb2gray(img)));
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
    std::pair<cv::Mat, std::vector<bool>> homography = findHomography(trainIDs, queryIDs,
        _features.back().getKeypoints(), feature.getKeypoints());
    const cv::Mat &H = homography.first;
    const std::vector<bool> &state = homography.second;

    // save the results
    _imgs.push_back(img.clone());
    _features.push_back(feature);
    _matches.push_back(match);
    _status.push_back(state);
    _homographies.push_back(H);
    _lastPano = _calculatePano();
}

const cv::Mat Stitcher::pano() const
{
    return _lastPano.first;
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

std::pair<cv::Mat, cv::Point> Stitcher::panoWithOrigin() const
{
    return _lastPano;
}

cv::Mat Stitcher::lastImg() const
{
    return _imgs.back().clone();
}
