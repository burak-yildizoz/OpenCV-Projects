#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>

#include "Stitch.hpp"
#include "imgops.hpp"
#include "general.hpp"

int main(int argc, char *argv[])
{
    const std::string prepath = (argc > 1) ? argv[1] : "jungle/";
    const std::string postpath = (argc > 2) ? argv[2] : ".jpg";

    auto filename = [&prepath, &postpath](int num) -> std::string {
        return prepath + std::to_string(num) + postpath;
    };

    auto images = [&filename](int num) -> cv::Mat {
        cv::Mat img = cv::imread(filename(num), cv::IMREAD_COLOR);
        CHECK(!img.empty());
        return imgops::resize(img, 480);
    };

    // pano: ... 1 -> 2
    Stitcher sfront(images(1));
    cv::Mat patchImg = images(2);
    // add until patch image
    sfront.add(patchImg);

    // pano: ... 3 -> 2
    Stitcher sback(images(3));
    // add until patch image
    sback.add(patchImg);

    // merge: ... 1 -> 2 <- 3 ...
    std::pair<cv::Mat, cv::Point> patchedPano = Stitcher::patchPano(
        sfront.panoWithOrigin(), sback.panoWithOrigin());
    const cv::Mat &pano = patchedPano.first;

    cv::imshow("first stitch", sfront.pano());
    cv::imshow("second stitch", sback.pano());
    cv::imshow("pano", pano);
    cv::waitKey();

    // warp: ... 1 -> 2 <- 3 ...
    //                |
    //                v
    //                4
    cv::Mat nextImg = images(4);
    Stitcher scont(nextImg, pano, cv::Rect(patchedPano.second, patchImg.size()));
    // continue adding

    cv::imshow("matches", scont.drawMatches());
    cv::imshow("stitching", scont.newestStitch());
    cv::imshow("new pano", scont.pano());
    cv::waitKey();

    cv::destroyAllWindows();
    return 0;
}
