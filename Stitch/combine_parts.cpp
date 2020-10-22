#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>

#include "Stitch.hpp"
#include "imgops.hpp"
#include "general.hpp"

int main(int argc, char *argv[])
{
    const std::string prepath = (argc > 1) ? argv[1] : "jungle/";
    const std::string postpath = (argc > 2) ? argv[2] : ".jpg";

    auto filename = [&prepath, &postpath](int num) -> std::string {
        return prepath + std::to_string(num) + postpath;
    };

    auto images = [&filename](int num) -> cv::Mat {
        cv::Mat img = cv::imread(filename(num), cv::IMREAD_COLOR);
        CHECK(!img.empty());
        return imgops::resize(img, 480);
    };

    const char winname[] = "map";
    cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);

    // create a map where visited points are true
    cv::Mat local_map = (cv::Mat_<uchar>(3, 6) <<
                         0, 0, 1, 1, 1, 1,
                         0, 1, 1, 1, 1, 0,
                         1, 1, 1, 1, 0, 0) * 255;
    cv::imshow(winname, local_map);
    std::cout << "Resize the window to see the local map." << std::endl;
    cv::waitKey();

    // provide the function that given x and y coordinates
    // return the image at that location in local map
    auto Im = std::function<cv::Mat(int, int)>([&local_map, &images](int x, int y) -> cv::Mat {
        int num = x + local_map.cols * y + 1;
        return images(num);
    });

    // combine the images in the map
    cv::Mat combined = Stitcher::combineImages(local_map, Im);
    cv::imshow(winname, combined);
    cv::waitKey();

    cv::destroyAllWindows();
    return 0;
}
