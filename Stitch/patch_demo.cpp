#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

#include <Stitch/Stitch.hpp>
#include <general/general.hpp>
#include <general/imgops.hpp>

int main(int argc, char *argv[]) {
  const std::string keys = Appender::keys;
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Demonstration of combining 4 images\n"
               "0 -> 1 <- 2\n"
               "     |\n"
               "     v\n"
               "     3");
  if ((argc == 1) || parser.has("help"))
    parser.printMessage();

  const std::string path = parser.get<std::string>("@path");
  const bool use_affine = parser.has("affine");
  std::cout << "Stitcher mode: " << (use_affine ? "Affine" : "Perspective")
            << std::endl;

  auto filename = [&path](int num) -> std::string {
    return general::string_format(path, num);
  };
  DEBUG(filename(0));

  auto images = [&filename](int num) -> cv::Mat {
    cv::Mat img = cv::imread(filename(num), cv::IMREAD_COLOR);
    CHECK(!img.empty());
    return imgops::resize(img, 480);
  };

  // pano: ... 0 -> 1
  std::shared_ptr<Stitcher> sfront = Appender::create(use_affine, images(0));
  cv::Mat patchImg = images(1);
  // add until patch image
  sfront->add(patchImg);

  // pano: ... 2 -> 1
  std::shared_ptr<Stitcher> sback = Appender::create(use_affine, images(2));
  // add until patch image
  sback->add(patchImg);

  // merge: ... 0 -> 1 <- 2 ...
  std::pair<cv::Mat, cv::Point> patchedPano =
      Stitcher::patchPano(sfront->panoWithOrigin(), sback->panoWithOrigin());
  const cv::Mat &pano = patchedPano.first;

  auto show_resizable = [](std::string winname, const cv::Mat &img) {
    cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);
    cv::imshow(winname, img);
  };

  show_resizable("first stitch", sfront->pano());
  show_resizable("second stitch", sback->pano());
  show_resizable("pano", pano);
  cv::waitKey();

  // warp: ... 0 -> 1 <- 2 ...
  //                |
  //                v
  //                3
  cv::Mat nextImg = images(3);
  std::shared_ptr<Stitcher> scont = Appender::create(
      use_affine, nextImg, pano, cv::Rect(patchedPano.second, patchImg.size()));
  // continue adding

  show_resizable("matches", scont->drawMatches());
  show_resizable("stitching", scont->newestStitch());
  show_resizable("new pano", scont->pano());
  cv::waitKey();

  cv::destroyAllWindows();
  return 0;
}
