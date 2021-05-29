#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

#include <Stitch/Stitch.hpp>
#include <general/general.hpp>
#include <general/imgops.hpp>

int main(int argc, char *argv[]) {
  const std::string keys = Appender::keys +
                           "{start first | 0    | First image index }"
                           "{end last    | 1000 | Last image index }";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Stitch images in given range");
  if ((argc == 1) || parser.has("help"))
    parser.printMessage();

  const std::string path = parser.get<std::string>("@path");
  const bool use_affine = parser.has("affine");
  std::cout << "Stitcher mode: " << (use_affine ? "Affine" : "Perspective")
            << std::endl;
  const int start = parser.get<int>("start");
  const int end = parser.get<int>("end");

  auto filename = [&path](int num) -> std::string {
    return general::string_format(path, num);
  };
  DEBUG(filename(start));

  auto images = [&filename](int num) -> cv::Mat {
    cv::Mat img = cv::imread(filename(num), cv::IMREAD_COLOR);
    CHECK(!img.empty());
    return imgops::resize(img, 480);
  };

  cv::Mat img = images(start);

  std::cout << "Press any key to continue" << std::endl;
  std::cout << "Press space to pause" << std::endl;
  std::cout << "Press ESC to exit" << std::endl;

  const char winname[] = "pano";
  cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);
  cv::imshow(winname, img);
  char ch = cv::waitKey();

  std::shared_ptr<Stitcher> stitcher = Appender::create(use_affine, img);
  // press ESC to exit
  for (int i = start + 1; (ch != 27) && (i <= end); i++) {
    std::cout << i << std::endl;
    // read next image and resize
    img = images(i);
    // do stitching and get the results
    stitcher->add(img);
    cv::Mat matches = stitcher->drawMatches();
    cv::Mat stitching = stitcher->newestStitch();
    cv::Mat pano = stitcher->pano();
    // show results
    cv::imshow("matches", matches);
    cv::imshow("stitching", stitching);
    cv::imshow(winname, pano);
    ch = cv::waitKey(1);
    // wait for user if there may be a problem
    static int lastArea = pano.size().area();
    if (pano.size().area() > 1.4 * lastArea) {
      std::cout << "Check if there is any problem!" << std::endl;
      ch = cv::waitKey();
      std::cout << "Checked!" << std::endl;
    }
    // press space key to pause
    if (ch == ' ') {
      std::cout << "Paused!" << std::endl;
      ch = cv::waitKey();
      std::cout << "Continued!" << std::endl;
    }
  }

  cv::destroyAllWindows();
  return 0;
}
