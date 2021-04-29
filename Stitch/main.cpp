#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#include <Stitch/Stitch.hpp>
#include <general/general.hpp>
#include <general/imgops.hpp>

int main(int argc, char *argv[]) {
  const std::string prepath = (argc > 1) ? argv[1] : "frame_";
  const std::string postpath = (argc > 2) ? argv[2] : ".jpg";

  const int start = (argc > 3) ? atoi(argv[3]) : 1;
  const int end = (argc > 4) ? atoi(argv[4]) : 332;

  auto filename = [&prepath, &postpath](int num) -> std::string {
    return prepath + std::to_string(num) + postpath;
  };
  DEBUG(filename(0));

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

  Stitcher stitcher(img);
  // press ESC to exit
  for (int i = start + 1; (ch != 27) && (i <= end); i++) {
    std::cout << i << std::endl;
    // read next image and resize
    img = images(i);
    // do stitching and get the results
    stitcher.add(img);
    cv::Mat matches = stitcher.drawMatches();
    cv::Mat stitching = stitcher.newestStitch();
    cv::Mat pano = stitcher.pano();
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
