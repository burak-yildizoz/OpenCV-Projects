#include "Calibrate/Fisheye.hpp"
#include <general/general.hpp>
#include <iostream>
#include <memory>

const std::string keys =
    "{help h usage ? |   | print this message }"
    "{num_imgs n     | 5 | number of chessboard images }"
    "{@img_stream    | ../data/fisheye/left%02d.jpg | calculate intrinsics "
    "from given data }"
    "{@res_stream    |   | display results on test data }"
    "{pinhole p      |   | use pinhole model (default is fisheye) }";

int main(int argc, char *argv[]) {
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Camera Calibration");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  const int num_imgs = parser.get<int>("num_imgs");
  std::string img_stream = parser.get<std::string>("@img_stream");
  std::string res_stream = parser.get<std::string>("@res_stream");
  if (res_stream.empty())
    res_stream = img_stream;
  const bool ispinhole = parser.has("pinhole");
  ;

  std::cout << "Using camera model: " << (ispinhole ? "pinhole" : "fisheye")
            << std::endl;
  std::unique_ptr<Calibrate> cal = ispinhole
                                       ? std::make_unique<Calibrate>(num_imgs)
                                       : std::make_unique<Fisheye>(num_imgs);

  cv::VideoCapture cap;
  int delay; // waitKey(delay)

  auto setCap = [&cap, &delay](const std::string &stream) {
    if (stream.size() == 1) { // open camera
      std::cout << "Camera index " << stream << std::endl;
      cap.open(atoi(stream.c_str()));
      delay = 1;
    } else { // re-read from saved images
      std::cout << "Reading from " << stream << std::endl;
      cap.open(stream);
      delay = 0;
    }
  };

  setCap(img_stream);
  CHECK(cap.isOpened());

  // save object & image points to external files
  CHECK(cal->save_chessboard(cap, delay));

  cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
              static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
  if (sz.empty()) {
    std::cout << "Image size is not determined from VideoCapture" << std::endl;
    std::cout << "Using the size of the first image" << std::endl;
    CHECK(cap.set(cv::CAP_PROP_POS_FRAMES, 0));
    cv::Mat img;
    CHECK(cap.read(img));
    sz = img.size();
  }

  // read object & image points from external files
  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<std::vector<cv::Point2f>> image_points;
  CHECK(cal->load_chessboard(object_points, image_points));

  // get camera matrix and distortion coefficients
  cal->calibrate_camera(object_points, image_points, sz);

  // display the results
  cap.release();

  setCap(res_stream);
  CHECK(cap.isOpened());

  cal->display_undistorted_all(cap, delay);

  return 0;
}
