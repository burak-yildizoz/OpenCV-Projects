// standard library
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
// OpenCV libraries
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Segmenter.hpp"	// graph-based segmentation
#include "Stitch.hpp"	// feature matching
#include "imgops.hpp"	// image operations such as blend, rgb2gray, etc.
#include "general.hpp"	// handy tools such as check, debug, etc.

#include <cstdlib>

int main(int argc, char **argv)
{
    const std::string path = (argc > 1) ? argv[1] : "../dataset/car/%04d.jpg";
    const int cam_id = atoi(path.c_str());

    // create a colormap for displaying purposes
    std::vector<cv::Vec3b> rainbow = imgops::colormap(cv::COLORMAP_RAINBOW, true);

    cv::VideoCapture cap;
    if (cam_id || path == "0")
        cap.open(cam_id);
    else
        cap.open(path);
    CHECK(cap.isOpened());
    cv::Mat img;
    bool paused = false;

    while (true)
    {
        // read the image
        cap >> img;
        if (img.empty())
            break;
        img = imgops::resize(img, 400);
        if (img.channels() == 1)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        // perform graph-based segmentation
        Segmenter segment(img);
        DEBUG(segment.numSegments);

        // detect features in the image
        Feature feature(imgops::rgb2gray(img));

        // match the features between frames
        static cv::Mat lastImg = img;
        Feature lastFeature(imgops::rgb2gray(lastImg));
        std::pair<std::vector<int>, std::vector<int>> matches = Stitcher::findMatch(lastFeature.desc, feature.desc);
        std::vector<bool> status = Stitcher::findHomography(matches.first, matches.second, lastFeature.getKeypoints(), feature.getKeypoints()).second;
        std::vector<int> trainIDs, queryIDs;
        trainIDs.reserve(status.size());
        queryIDs.reserve(status.size());
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i])
            {
                trainIDs.push_back(matches.first[i]);
                queryIDs.push_back(matches.second[i]);
            }
        }

        // find out which segments are matched
        Segmenter lastSegment(lastImg);
        DEBUG(lastSegment.numSegments);
        std::vector< std::vector<int> > featuresOfSegments(segment.numSegments);
        std::vector< std::vector<int> > featuresOfPrevSegments(lastSegment.numSegments);
        for (size_t i = 0; i < queryIDs.size(); i++)
        {
            int featureID = queryIDs[i];
            cv::Point2f kp = feature.kps[featureID].pt;
            int segmentID = segment.belongs(kp);
            int prevFeatureID = trainIDs[i];
            cv::Point2f prev_kp = lastFeature.kps[prevFeatureID].pt;
            int prevSegmentID = lastSegment.belongs(prev_kp);
            featuresOfSegments[segmentID].push_back(prevSegmentID);
            featuresOfPrevSegments[prevSegmentID].push_back(segmentID);
        }

        // decide segment matching
        std::vector<int> prevIDs, currIDs;
        for (size_t i = 0; i < featuresOfSegments.size(); i++)
        {
            // if the segment has matched any segment from previous image
            if (featuresOfSegments[i].size())
            {
                std::pair<int,int> occurrence = general::most_frequent_element(featuresOfSegments[i]);
                int prevID = occurrence.first;
                // make sure it is also matched the other way around
                std::pair<int, int> lastOccurrence = general::most_frequent_element(featuresOfPrevSegments[prevID]);
                int nextID = lastOccurrence.first;
                if ((int)i == nextID)
                {
                    prevIDs.push_back(prevID);
                    currIDs.push_back(i);
                    int count = occurrence.second;
                    size_t total = featuresOfSegments[i].size();
                    printf("%zu is matched with %d with certainity %d/%zu\n", i, prevID, count, total);
                }
                else
                {
                    // printf("%zu is matched with %d but %d matches with %d\n", i, prevID, prevID, nextID);
                }
            }
        }
        DEBUG(prevIDs.size());	// number of matches

        // paint the matched segments
        static std::vector<cv::Vec3b> prevColors(rainbow.begin(), rainbow.begin() + lastSegment.numSegments);	// last label colors
        std::vector<bool> colorUsed(256, false);	// currently used colors
        std::vector<cv::Vec3b> currColors(segment.numSegments, cv::Vec3b(0, 0, 0));	// label colors
        // if there is a match, use the same color between segments
        for (size_t i = 0; i < prevIDs.size(); i++)
        {
            int prevID = prevIDs[i];
            int currID = currIDs[i];
            cv::Vec3b color = prevColors[prevID];
            currColors[currID] = color;
            colorUsed[std::distance(rainbow.begin(), std::find(rainbow.begin(), rainbow.end(), color))] = true;
        }
        // otherwise, pick currently unpicked colors
        for (cv::Vec3b& color : currColors)
        {
            if (color == cv::Vec3b(0, 0, 0))
            {
                size_t index = std::distance(colorUsed.begin(), std::find(colorUsed.begin(), colorUsed.end(), false));
                color = rainbow[index];
                colorUsed[index] = true;
            }
        }
        // colorize the label image
        cv::Mat disp(img.size(), CV_8UC3);
        cv::Mat labelImage = segment.getLabelImage();
        labelImage.forEach<uchar>([&currColors, &disp](uchar label, const int pos[2])
        {
            disp.at<cv::Vec3b>(pos) = currColors[label];
        });

        // show original images
        ConnectImages ci(lastImg.size(), img.size());
        cv::Mat compOrig = ci.compareImages(lastImg, img);

        // show feature matching results
        cv::Mat compFeature = ci.compareImages(lastImg, img);
        ci.connectCenters(compFeature, trainIDs, queryIDs, lastFeature.kps, feature.kps);

        // show colorized labels
        static cv::Mat lastDisp = disp;
        cv::Mat compLabels = ci.compareImages(lastDisp, disp);

        // show transparent matching segments with centers connected and numbered
        cv::Mat compTr = ci.compareImages(lastImg, img);
        cv::Mat painted = compTr.clone();
        std::vector<Contour> prevContours = lastSegment.contoursOfLabels();
        std::vector<Contour> currContours = segment.contoursOfLabels();
        ci.drawContours(painted, prevContours, currContours, prevIDs, currIDs, prevColors, currColors);
        compTr = imgops::blend(compTr, painted, 0.3);
        ci.drawContours(compTr, prevContours, currContours, prevIDs, currIDs, prevColors, currColors, 1, false);
        ci.connectCenters(compTr, prevContours, currContours, prevIDs, currIDs, currColors, false);
        ci.numberPoints(compTr, prevContours, currContours, prevIDs, currIDs,
            prevColors, currColors, CV_RGB(0, 255, 0));

        // show results
        cv::imshow("original", compOrig);
        cv::imshow("feature matching", compFeature);
        cv::imshow("segmentation", compLabels);
        cv::imshow("matched segments", compTr);
        char ch = cv::waitKey(paused ? 0 : 1);
        if (ch == ' ')
            paused = !paused;
        if (ch == 27)	// ESC
            break;

        // update variables
        lastFeature = feature;
        lastImg = img;
        prevColors = currColors;
        lastDisp = disp;

        std::cout << std::endl;
    }

    return 0;
}
