// OpenCV includes
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
// handy tools such as DEBUG, CHECK, etc.
#include "general.hpp"
// header file
#include "Calibrate.hpp"

template <class T>
void Calibrate::write_to_file(std::ofstream &ofs, const std::vector<std::vector<T>> &obj)
{
    CHECK(ofs.is_open());
    for (size_t i = 0; i < obj.size(); i++)
    {
        for (size_t j = 0; j < obj[i].size(); j++)
            ofs << obj[i][j] << "\t";
        ofs << "\n";
    }
    ofs.close();
}

Calibrate::Calibrate(int numBoards, int numCornersHor, int numCornersVer,
                     const std::string &object_filename, const std::string &image_filename)
    : _numBoards(numBoards), _numCornersHor(numCornersHor), _numCornersVer(numCornersVer),
      _object_filename(object_filename), _image_filename(image_filename)
{
    CHECK(numBoards > 0);
    CHECK(numCornersHor > 0);
    CHECK(numCornersVer > 0);
    CHECK(numBoards * numCornersHor * numCornersVer >= 6);
    CHECK(!object_filename.empty());
    CHECK(!image_filename.empty());
}

int Calibrate::save_chessboard(cv::VideoCapture *capture, int delay, std::string image_name, std::string ext) const
{
    cv::VideoCapture &cap = *capture;
    // make sure image stream is open
    CHECK(cap.isOpened());
    cv::Mat img;
    cap >> img;
    CHECK(!img.empty());

    std::cout << "Press space key to save the chessboard properties\n"
                 "Press S to save also the image\n"
                 "Press ESC to cancel and quit\n"
              << std::endl;

    // create window for display purposes
    const char *winname = "image";
    cv::namedWindow(winname);

    // required for object points
    int numSquares = numCornersHor * numCornersVer;
    // the chessboard is assumed to be on the xy-plane
    // and the physical length of the edges does not really matter
    // (0,0,0), (0,1,0), (0,2,0), ..., (0,5,8)
    // here, the length of the square is assumed to be 1mm
    std::vector<cv::Point3f> obj;
    for (int j = 0; j < numSquares; j++)
        obj.push_back(cv::Point3f(j / numCornersHor, j % numCornersHor, 0.0f));

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<cv::Point2f> corners;
    const cv::Size board_sz(numCornersHor, numCornersVer);
    int successes = 0;

    while (successes < numBoards)
    {
        if (img.empty())
        {
            std::cerr << "The captured image is empty!" << std::endl;
            cv::destroyWindow(winname);
            return false;
        }

        // check if image contains chessboard
        bool found = findChessboardCorners(img, board_sz, corners,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        cv::Mat disp_img = img.clone();
        if (found)
        {
            cv::Mat gray;
            cvtColor(img, gray, CV_BGR2GRAY);
            cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(disp_img, board_sz, corners, found);
        }

        // display the frame and chessboard corners
        imshow(winname, disp_img);
        char key = cv::waitKey(delay);

        if (key == 27) // ESC is pressed
        {
            std::cout << "save_chessboard cancelled!" << std::endl;
            cv::destroyWindow(winname);
            return false;
        }
        else if (found && (key == ' ' || std::tolower(key) == 's'))
        {
            image_points.push_back(corners);
            object_points.push_back(obj);

            successes++;
            printf("Snap #%d stored!\n", successes);
            if (std::tolower(key) == 's')
            {
                std::string _image_name = image_name + std::to_string(successes) + ext;
                bool res = cv::imwrite(_image_name, img);
                if (res)
                    std::cout << "Saved " << _image_name << std::endl;
            }
        }

        cap >> img;
    }
    cv::destroyWindow(winname);

    write_to_file(*new std::ofstream(object_filename), object_points);
    std::cout << "Saved object points to " << object_filename << std::endl;

    write_to_file(*new std::ofstream(image_filename), image_points);
    std::cout << "Saved image points to " << image_filename << std::endl;

    return successes;
}

int Calibrate::save_chessboard(const std::vector<cv::Mat> &chessboard_images, bool display, int delay) const
{
    std::cout << "Press ESC to cancel and quit\n"
              << std::endl;

    // create window for display purposes
    const char *winname = "image";
    if (display)
        cv::namedWindow(winname);

    // required for object points
    int numSquares = numCornersHor * numCornersVer;
    // the chessboard is assumed to be on the xy-plane
    // and the physical length of the edges does not really matter
    // (0,0,0), (0,1,0), (0,2,0), ..., (0,5,8)
    // here, the length of the square is assumed to be 1mm
    std::vector<cv::Point3f> obj;
    for (int j = 0; j < numSquares; j++)
        obj.push_back(cv::Point3f(j / numCornersHor, j % numCornersHor, 0.0f));

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<cv::Point2f> corners;
    const cv::Size board_sz(numCornersHor, numCornersVer);
    int successes = 0;

    for (size_t i = 0; i < chessboard_images.size(); i++)
    {
        const cv::Mat &img = chessboard_images[i];
        CHECK(!img.empty());

        // check if image contains chessboard
        bool found = findChessboardCorners(img, board_sz, corners,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        if (found)
        {
            cv::Mat gray;
            cvtColor(img, gray, CV_BGR2GRAY);
            cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
        }

        // display the frame and chessboard corners
        if (display)
        {
            cv::Mat disp_img = img.clone();
            drawChessboardCorners(disp_img, board_sz, corners, found);
            imshow(winname, disp_img);
            char key = cv::waitKey(delay);
            if (key == 27) // ESC is pressed
            {
                std::cout << "save_chessboard cancelled!" << std::endl;
                cv::destroyWindow(winname);
                return false;
            }
        }

        // add the results to the save list
        if (found)
        {
            image_points.push_back(corners);
            object_points.push_back(obj);
            successes++;
        }
    }
    if (display)
        cv::destroyWindow(winname);

    std::cout << successes << " chessboard images have been found!" << std::endl;
    if (successes <= 0)
    {
        std::cerr << "Nothing to save! Quitting..." << std::endl;
        return false;
    }
    else if (successes < numBoards)
    {
        std::cerr << "Warning: The number of boards specified in the construction of this object was "
                  << numBoards << std::endl;
    }

    write_to_file(*new std::ofstream(object_filename), object_points);
    std::cout << "Saved object points to " << object_filename << std::endl;

    write_to_file(*new std::ofstream(image_filename), image_points);
    std::cout << "Saved image points to " << image_filename << std::endl;

    return successes;
}

bool Calibrate::load_chessboard(std::vector<std::vector<cv::Point3f>> &object_points,
                                std::vector<std::vector<cv::Point2f>> &image_points) const
{
    // set the size of object_points and image_points
    int numSquares = numCornersHor * numCornersVer;
    object_points.resize(numBoards);
    for (size_t i = 0; i < object_points.size(); i++)
        object_points[i].resize(numSquares);
    image_points.resize(numBoards);
    for (size_t i = 0; i < image_points.size(); i++)
        image_points[i].resize(numSquares);

    bool res = false;

    // load parameters for object_points
    std::ifstream obj_ifs(object_filename);
    if (!obj_ifs.is_open())
    {
        std::cerr << "Could not open " << object_filename << std::endl;
        return false;
    }
    for (size_t i = 0; i < object_points.size(); i++)
    {
        for (size_t j = 0; j < object_points[i].size(); j++)
        {
            cv::Point3f &p = object_points[i][j];
            obj_ifs >> p; // overloaded in general.cpp
            if (!res && p != cv::Point3f())
                res = true;
            CHECK(obj_ifs.good());
        }
    }
    obj_ifs.close();
    if (!res)
    {
        std::cerr << "Could not load object points!" << std::endl;
        return false;
    }

    res = false;

    // load parameters for image_points
    std::ifstream img_ifs(image_filename);
    if (!img_ifs.is_open())
    {
        std::cerr << "Could not open " << image_filename << std::endl;
        return false;
    }
    for (size_t i = 0; i < image_points.size(); i++)
    {
        for (size_t j = 0; j < image_points[i].size(); j++)
        {
            cv::Point2f &p = image_points[i][j];
            img_ifs >> p; // overloaded in general.cpp
            if (!res && p != cv::Point2f())
                res = true;
        }
    }
    img_ifs.close();
    if (!res)
    {
        std::cerr << "Could not load image points!" << std::endl;
        return false;
    }

    return res;
}

void Calibrate::calibrate_camera(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                                 const std::vector<std::vector<cv::Point2f>> &imagePoints,
                                 cv::Size imageSize,
                                 cv::Mat &cameraMatrix,
                                 cv::Mat &distCoeffs)
{
    std::vector<cv::Mat> rvecs, tvecs;
    if (cameraMatrix.empty())
    {
        /*
            [fx 0  cx
              0 fy cy
              0  0  1]
        */
        cameraMatrix = cv::Mat(3, 3, CV_32FC1);
        cameraMatrix.ptr<float>(0)[0] = 1; // fx
        cameraMatrix.ptr<float>(1)[1] = 1; // fy
    }
    std::cout << "Calibrating!" << std::endl;
    calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    std::cout << "Calibration done!" << std::endl;
}

void Calibrate::display_undistorted(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                    cv::Mat &img, std::string winname)
{
    CHECK(!img.empty());
    bool show_once = winname.empty();
    if (show_once)
        winname = "undistorted";
    cv::Mat imageUndistorted;
    undistort(img, imageUndistorted, cameraMatrix, distCoeffs);
    imshow(winname, imageUndistorted);
    if (show_once)
    {
        char ch = cv::waitKey(0);
        CHECK(ch != 27);
        img = imageUndistorted.clone();
    }
}

void Calibrate::display_undistorted(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                                    cv::VideoCapture *capture, std::string winname)
{
    cv::VideoCapture &cap = *capture;
    // make sure image stream is open
    CHECK(cap.isOpened());
    cv::Mat img;
    cap >> img;
    CHECK(!img.empty());
    CHECK(!winname.empty());

    // for display purposes
    cv::namedWindow(winname);

    char ch = 0;
    while (ch != 27) // ESC is pressed
    {
        if (img.empty())
        {
            std::cout << "End of image stream!" << std::endl;
            break;
        }
        display_undistorted(cameraMatrix, distCoeffs, img, winname);
        ch = cv::waitKey(1);
        cap >> img;
    }
    cv::destroyWindow(winname);
}
