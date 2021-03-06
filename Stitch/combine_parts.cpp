#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

#include <Stitch/Stitch.hpp>
#include <general/general.hpp>
#include <general/imgops.hpp>

int main(int argc, char *argv[]) {
  const std::string keys =
      Appender::keys +
      "{@matpath | | path to local map saved in binary mode (see "
      "general::matwrite)\n"
      "example:\n"
      "[ 0   0  255 255 255 255]\n"
      "[ 0  255 255 255 255  0 ]\n"
      "[255 255 255 255  0   0 ]\n"
      "Image numbers 2-5, 7-10, 12-15 will be stitched in this example. }";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Demonstration of combining images given local map");
  if ((argc == 1) || parser.has("help"))
    parser.printMessage();

  const std::string path = parser.get<std::string>("@path");
  const bool use_affine = parser.has("affine");
  std::cout << "Stitcher mode: " << (use_affine ? "Affine" : "Perspective")
            << std::endl;
  const std::string matpath = parser.get<std::string>("@matpath");

  auto filename = [&path](int num) -> std::string {
    return general::string_format(path, num);
  };
  DEBUG(filename(0));

  auto images = [&filename](int num) -> cv::Mat {
    cv::Mat img = cv::imread(filename(num), cv::IMREAD_COLOR);
    CHECK(!img.empty());
    return imgops::resize(img, 480);
  };

  const char winname[] = "map";
  cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);

  // create a map where visited points are true
  // the default is a test local map for images obtained by PictureByParts
  cv::Mat local_map = (cv::Mat_<uchar>(3, 6) << 0, 0, 1, 1, 1, 1, //
                       0, 1, 1, 1, 1, 0,                          //
                       1, 1, 1, 1, 0, 0) *
                      255;
  if (!matpath.empty())
    local_map = general::matread(matpath);
  CHECK(!local_map.empty());
  cv::imshow(winname, local_map);
  std::cout << local_map.size() << std::endl;
  std::cout << "Resize the window to see the local map." << std::endl;
  cv::waitKey();
  std::cout << "Processing..." << std::endl;

  // provide the function that given x and y coordinates
  // return the image at that location in local map
  auto Im = std::function<cv::Mat(int, int)>(
      [&local_map, &images](int x, int y) -> cv::Mat {
        int num = x + local_map.cols * y;
        return images(num);
      });

  // combine the images in the map
  cv::Mat combined = Stitcher::combineImages(local_map, Im);
  cv::imshow(winname, combined);
  std::cout << "Here is the resulting image." << std::endl;
  cv::waitKey();

  return 0;
}
