#include "Fisheye.hpp"
#include "general.hpp"

int main(int argc, char *argv[])
{
    // number of chessboard images
    const int num_imgs = (argc > 1) ? atoi(argv[1]) : 5;
    // calculate intrinsics from given data
    // example: samples/data/left%02d.jpg
    const char *img_stream = (argc > 2) ? argv[2] : "../data/fisheye/left%2d.jpg";
    // display results on test data
    const char *res_stream = (argc > 3) ? argv[3] : "0";

    Fisheye cal(num_imgs);

    cv::VideoCapture cap;
    int delay; // waitKey(delay)

    auto setCap = [&cap, &delay](const char *stream) {
        if (strlen(stream) == 1)
        { // open camera
            cap.open(atoi(stream));
            delay = 1;
        }
        else
        { // re-read from saved images
            cap.open(stream);
            delay = 0;
        }
    };

    setCap(img_stream);
    CHECK(cap.isOpened());

    // save object & image points to external files
    CHECK(cal.save_chessboard(cap, delay));

    cv::Size sz(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // read object & image points from external files
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    CHECK(cal.load_chessboard(object_points, image_points));

    // get camera matrix and distortion coefficients
    cal.calibrate_camera(object_points, image_points, sz);

    // display the results
    // display the results
    cap.release();

    if (strcmp(img_stream, res_stream) != 0)
        setCap(res_stream);
    else
        setCap(img_stream);
    CHECK(cap.isOpened());

    cal.display_undistorted(cap);

    return 0;
}
