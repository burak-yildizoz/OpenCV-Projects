#include "Calibrate.hpp"
#include "general.hpp"

//#define NO_CAP
#ifdef NO_CAP
#include <opencv2/imgcodecs.hpp>
#endif // NO_CAP

int main()
{
	const int num_imgs = 5;
	// number of chessboard images: 5
	Calibrate cal(num_imgs);

#ifndef NO_CAP

	// open camera
	cv::VideoCapture cap(0);
	CHECK(cap.isOpened());

	// save object & image points to external files
	//CHECK(cal.save_chessboard(&cap));
	CHECK(cal.save_chessboard(new cv::VideoCapture("../data/pinhole/image%1d.jpg"), 0)); // re-read from saved images

	cv::Size sz(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

#else

	cv::Size sz;

	// read images
	std::vector<cv::Mat> imgs(num_imgs);
	for (int i = 0; i < num_imgs; i++)
	{
		std::string filename = "../data/pinhole/image" + std::to_string(i + 1) + ".jpg";
		imgs[i] = cv::imread(filename);
		CHECK(!imgs[i].empty());
		cv::Size size(imgs[0].cols, imgs[0].rows);
		if (i == 0)
			sz = size;
		else
			CHECK(sz == size);
	}

	// save object & image points to external files
	CHECK(cal.save_chessboard(imgs, true));

#endif // NO_CAP

	// read object & image points from external files
	std::vector<std::vector<cv::Point3f>> object_points;
	std::vector<std::vector<cv::Point2f>> image_points;
	CHECK(cal.load_chessboard(object_points, image_points));

	// get camera matrix and distortion coefficients
	cv::Mat intrinsics, distCoeffs;
	cal.calibrate_camera(object_points, image_points, sz, intrinsics, distCoeffs);

	// display the results
#ifndef NO_CAP
	cal.display_undistorted(intrinsics, distCoeffs, &cap);
#else
	for (int i = 0; i < num_imgs; i++)
		cal.display_undistorted(intrinsics, distCoeffs, imgs[i]);
#endif // NO_CAP

	return 0;
}
