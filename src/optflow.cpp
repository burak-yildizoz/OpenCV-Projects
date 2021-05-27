#include <general/general.hpp>
#include <general/imgops.hpp>
#include <opencv2/highgui.hpp>

const std::string keys = "{ optflow t | FBACK      | optical flow type }"
                         "{ input i   | vtest.avi  | image stream      }";

int main(int argc, char **argv) {
  // get input and open image stream
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("OpenCV Optical Flow API");
  parser.printMessage();
  std::string optflowType = parser.get<std::string>("optflow");
  cv::Ptr<cv::DenseOpticalFlow> optflow;
  bool use_rgb;
  try {
    optflow = imgops::get_optflow(optflowType, use_rgb);
  } catch (const std::runtime_error &) {
    std::cout << "Invalid optical flow type!" << std::endl;
    DEBUG(imgops::get_optflow_types());
    return EXIT_FAILURE;
  }
  std::string input = parser.get<std::string>("input");
  DEBUG(input);
  cv::VideoCapture cap;
  CHECK(general::videocapture_open(cap, input));
  // calculate dense optical flow on each frame
  cv::Mat last_img;
  CHECK(cap.read(last_img));
  const std::string winname = "Dense Optical Flow";
  cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);
  bool paused = false;
  bool show_vectors = false;
  while (true) {
    // read the next frame
    cv::Mat img;
    if (!cap.read(img)) {
      std::cout << "End of image stream!" << std::endl;
      break;
    }
    // measure runtime of dense optical flow
    double timer = static_cast<double>(cv::getTickCount());
    cv::Mat flow;
    if (use_rgb)
      optflow->calc(last_img, img, flow);
    else
      optflow->calc(imgops::bgr2gray(last_img), imgops::bgr2gray(img), flow);
    double fps = cv::getTickFrequency() / (cv::getTickCount() - timer);
    // display the result
    cv::Mat disp_img;
    if (show_vectors) {
      disp_img = img.clone();
      imgops::drawOptFlowMap(flow, disp_img);
    } else {
      disp_img = imgops::reprOptFlow(flow);
    }
    // display frame rate
    double scale = 0.75;
    cv::Scalar disp_color = CV_RGB(0, 255, 255);
    int thickness = 2;
    std::string fps_str = general::string_format("FPS: %.f", fps);
    cv::putText(disp_img, fps_str, cv::Point(100, 50), cv::FONT_HERSHEY_SIMPLEX,
                scale, disp_color, thickness);
    cv::imshow(winname, disp_img);
    char ch = cv::waitKey(paused ? 0 : 1);
    if (ch == 27) // press ESC to exit
      break;
    if (ch == ' ') // press SPACE to toggle pause
      paused = !paused;
    if (std::tolower(ch) ==
        'v') // press 'V' to toggle optical flow representation
      show_vectors = !show_vectors;
    last_img = img;
  }
  return 0;
}
