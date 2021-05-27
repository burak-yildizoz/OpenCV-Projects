#include <general/general.hpp>
#include <general/imgops.hpp>
#include <opencv2/highgui.hpp>

const std::string keys = "{ tracker t | CSRT      | tracker type }"
                         "{ input i   | vtest.avi | image stream }";

int main(int argc, char **argv) {
  // get input and open image stream
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("OpenCV Tracker API");
  parser.printMessage();
  std::string trackerType = parser.get<std::string>("tracker");
  cv::Ptr<cv::Tracker> tracker;
  try {
    tracker = imgops::get_tracker(trackerType);
  } catch (const std::runtime_error &) {
    std::cout << "Invalid tracker type!" << std::endl;
    DEBUG(imgops::get_tracker_types());
    return EXIT_FAILURE;
  }
  std::string input = parser.get<std::string>("input");
  DEBUG(input);
  cv::VideoCapture cap;
  CHECK(general::videocapture_open(cap, input));
  // select the initial bounding box to track
  cv::Mat img;
  CHECK(cap.read(img));
  const std::string winname = trackerType + " Tracker";
  cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);
  cv::Rect bbox;
  auto select_box = [&img, &winname, &bbox, &tracker]() {
    bool showCrosshair = false;
    bbox = cv::selectROI(winname, img, showCrosshair);
    tracker->init(img, bbox);
  };
  select_box();
  std::cout << "Initial bounding box: " << bbox << std::endl;
  // track the selected area on each frame
  bool paused = false;
  while (cap.read(img)) {
    // measure runtime of tracking
    double timer = static_cast<double>(cv::getTickCount());
    bool ok = tracker->update(img, bbox);
    double fps = cv::getTickFrequency() / (cv::getTickCount() - timer);
    // draw the result
    cv::Mat disp_img = img.clone();
    if (ok) {
      // Tracking success : Draw the tracked object
      cv::rectangle(disp_img, bbox, CV_RGB(0, 0, 255), 2);
    } else {
      cv::putText(disp_img, "Tracking failure detected", cv::Point(100, 80),
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(255, 0, 0), 2);
      cv::imshow(winname, disp_img);
      int ch = cv::waitKey();
      if (ch == 13) // press ENTER to manually select bounding box
      {
        select_box();
        disp_img = img.clone();
      } else if (ch != 27) // press ESC to cancel tracking
      {
        // press any other key to use the previous area
        tracker->init(img, bbox);
      }
    }
    // display the result
    double scale = 0.75;
    cv::Scalar disp_color = CV_RGB(0, 255, 255);
    int thickness = 2;
    // show tracker type
    std::string tracker_str = winname;
    cv::putText(disp_img, tracker_str, cv::Point(100, 20),
                cv::FONT_HERSHEY_SIMPLEX, scale, disp_color, thickness);
    // show frame rate
    std::string fps_str = general::string_format("FPS: %.f", fps);
    cv::putText(disp_img, fps_str, cv::Point(100, 50), cv::FONT_HERSHEY_SIMPLEX,
                scale, disp_color, thickness);
    // handle pressed key
    cv::imshow(winname, disp_img);
    char ch = cv::waitKey(paused ? 0 : 1);
    if (ch == ' ') // press SPACE to toggle pause
      paused = !paused;
    if (ch == 13) // press ENTER to re-select ROI
      select_box();
    if (ch == 27) // press ESC to exit
      break;
  }
  return 0;
}
