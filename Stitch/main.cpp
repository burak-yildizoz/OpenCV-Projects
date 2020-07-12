#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>

#include "Stitch.hpp"
#include "imgops.hpp"
#include "general.hpp"

int main()
{
    const std::string prepath = "frame_";
    const std::string postpath = ".jpg";

    int start = 1;
    int end = 332;

    auto filename = [&prepath, &postpath](int num) -> std::string
    {
        return prepath + std::to_string(num) + postpath;
    };

    auto images = [&filename](int num) -> cv::Mat
    {
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
    for (int i = start + 1; (ch != 27) && (i <= end); i++)
    {
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
        if (pano.size().area() > 1.4 * lastArea)
        {
            std::cout << "Check if there is any problem!" << std::endl;
            ch = cv::waitKey();
            std::cout << "Checked!" << std::endl;
        }
        // press space key to pause
        if (ch == ' ')
        {
            std::cout << "Paused!" << std::endl;
            ch = cv::waitKey();
            std::cout << "Continued!" << std::endl;
        }
    }

/*
    cv::destroyAllWindows();

    // pano: ... 1 -> 2
    Stitcher sfront(images(1));
    cv::Mat patchImg = images(2);
    // add until patch image
    sfront.add(patchImg);

    // pano: ... 3 -> 2
    Stitcher sback(images(3));
    // add until patch image
    sback.add(patchImg);

    // merge: ... 1 -> 2 <- 3 ...
    std::pair<cv::Mat, cv::Point> patchedPano = Stitcher::patchPano(
        sfront.panoWithOrigin(), sback.panoWithOrigin());
    const cv::Mat& pano = patchedPano.first;
    
    cv::imshow("first stitch", sfront.pano());
    cv::imshow("second stitch", sback.pano());
    cv::imshow("pano", pano);
    cv::waitKey();

    // warp: ... 1 -> 2 <- 3 ...
                      |
                      v
                      4
    cv::Mat nextImg = images(4);
    Stitcher scont(nextImg, pano, cv::Rect(patchedPano.second, patchImg.size()));
    // continue adding

    cv::imshow("matches", scont.drawMatches());
    cv::imshow("stitching", scont.newestStitch());
    cv::imshow("new pano", scont.pano());
    cv::waitKey();
*/
/*
    const char winname[] = "map";
    cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);

    // create a map where visited points are true
    cv::Mat local_map = (cv::Mat_<uchar>(3, 6) <<
            0, 0, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 0, 0) * 255;
    cv::imshow(winname, local_map);
    cv::waitKey();

    // provide the function that given x and y coordinates
    // return the image at that location in local map
    auto Im = std::function<cv::Mat(int, int)>([&local_map, &images](int x, int y) -> cv::Mat
    {
        int num = x + local_map.cols * y + 1;
        return images(num);
    });

    // combine the images in the map
    cv::Mat combined = Stitcher::combineImages(local_map, Im);
    cv::imshow(winname, combined);
    cv::waitKey();
*/
    cv::destroyAllWindows();
    return 0;
}
