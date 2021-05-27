#include <general/general.hpp>
#include <general/imgops.hpp>
#include <stdexcept>
#ifdef HAVE_OPENCV_OPTFLOW
#include <opencv2/optflow.hpp>
#endif
#ifdef HAVE_OPENCV_TRACKING
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#endif

namespace imgops {

std::vector<std::string> get_optflow_types() {
  return {"DIS", "FBACK"
#ifdef HAVE_OPENCV_OPTFLOW
          ,
          "RLOF", "TVL1", "PCA"
#endif
  };
}

// https://docs.opencv.org/master/df/dde/classcv_1_1DenseOpticalFlow.html
cv::Ptr<cv::DenseOpticalFlow> get_optflow(std::string optflowType,
                                          bool &use_rgb) {
  general::toupper(optflowType);
  use_rgb = false;
  if (optflowType == "DIS")
    return cv::DISOpticalFlow::create();
  if (optflowType == "FBACK")
    return cv::FarnebackOpticalFlow::create();
#ifdef HAVE_OPENCV_OPTFLOW
  if (optflowType == "RLOF") {
    use_rgb = true;
    return cv::optflow::DenseRLOFOpticalFlow::create();
  }
  if (optflowType == "TVL1")
    return cv::optflow::DualTVL1OpticalFlow::create();
  if (optflowType == "PCA")
    return cv::makePtr<cv::optflow::OpticalFlowPCAFlow>(
        cv::optflow::OpticalFlowPCAFlow());
#endif
  throw std::runtime_error("Unknown optical flow type.");
}

std::vector<std::string> get_tracker_types() {
  return {"MIL",
          "GOTURN"
#ifdef HAVE_OPENCV_TRACKING
          ,
          "CSRT",
          "KCF",
          "BOOSTING",
          "MEDIANFLOW",
          "TLD",
          "MOSSE"
#endif
  };
}

// https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/samples_utility.hpp
cv::Ptr<cv::Tracker> get_tracker(std::string trackerType) {
  general::toupper(trackerType);
  if (trackerType == "MIL")
    return cv::TrackerMIL::create();
  if (trackerType == "GOTURN")
    return cv::TrackerGOTURN::create();
#ifdef HAVE_OPENCV_TRACKING
  if (trackerType == "CSRT")
    return cv::TrackerCSRT::create();
  if (trackerType == "KCF")
    return cv::TrackerKCF::create();
  if (trackerType == "BOOSTING")
    return cv::legacy::upgradeTrackingAPI(
        cv::legacy::TrackerBoosting::create());
  if (trackerType == "MEDIANFLOW")
    return cv::legacy::upgradeTrackingAPI(
        cv::legacy::TrackerMedianFlow::create());
  if (trackerType == "TLD")
    return cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerTLD::create());
  if (trackerType == "MOSSE")
    return cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerMOSSE::create());
#endif
  throw std::runtime_error("Unknown tracker type.");
}

} // namespace imgops
