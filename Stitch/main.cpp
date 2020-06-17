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

	cv::Mat img = cv::imread(filename(start), cv::IMREAD_COLOR);
	CHECK(!img.empty());
	img = imgops::resize(img, 480);

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
		img = cv::imread(prepath + std::to_string(i) + postpath, cv::IMREAD_COLOR);
		CHECK(!img.empty());
		img = imgops::resize(img, 480);
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
		if (stitching.size().area() > 1.4 * img.size().area())
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
	
    cv::destroyAllWindows();
    return 0;
}
