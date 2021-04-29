#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

#include <Stitch/Stitch.hpp>
#include <Stitch/general.hpp>
#include <Stitch/imgops.hpp>

int main(int argc, char *argv[]) {
  const std::string prepath = (argc > 1) ? argv[1] : "jungle/";
  const std::string postpath = (argc > 2) ? argv[2] : ".jpg";

  auto filename = [&prepath, &postpath](int num) -> std::string {
    return prepath + std::to_string(num) + postpath;
  };
  DEBUG(filename(0));

  auto images = [&filename](int num) -> cv::Mat {
    cv::Mat img = cv::imread(filename(num), cv::IMREAD_COLOR);
    CHECK(!img.empty());
    return imgops::resize(img, 480);
  };

  // pano: ... 0 -> 1
  Stitcher sfront(images(0));
  cv::Mat patchImg = images(1);
  // add until patch image
  sfront.add(patchImg);

  // pano: ... 2 -> 1
  Stitcher sback(images(2));
  // add until patch image
  sback.add(patchImg);

  // merge: ... 0 -> 1 <- 2 ...
  std::pair<cv::Mat, cv::Point> patchedPano =
      Stitcher::patchPano(sfront.panoWithOrigin(), sback.panoWithOrigin());
  const cv::Mat &pano = patchedPano.first;

  cv::imshow("first stitch", sfront.pano());
  cv::imshow("second stitch", sback.pano());
  cv::imshow("pano", pano);
  cv::waitKey();

  // warp: ... 0 -> 1 <- 2 ...
  //                |
  //                v
  //                3
  cv::Mat nextImg = images(3);
  Stitcher scont(nextImg, pano, cv::Rect(patchedPano.second, patchImg.size()));
  // continue adding

  cv::imshow("matches", scont.drawMatches());
  cv::imshow("stitching", scont.newestStitch());
  cv::imshow("new pano", scont.pano());
  cv::waitKey();

  cv::destroyAllWindows();
  return 0;
}
