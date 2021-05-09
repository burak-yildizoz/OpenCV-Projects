// OpenCV includes
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
// standard library
#include <fstream>
// handy tools such as DEBUG, CHECK, etc.
#include "general.hpp"
// header file
#include "Calibrate.hpp"

template <class T>
void Calibrate::write_to_file(std::string filename, const std::vector<std::vector<T>> &obj)
{
    std::ofstream ofs(filename);
    CHECK(ofs.is_open());
    for (size_t i = 0; i < obj.size(); i++)
    {
        for (size_t j = 0; j < obj[i].size(); j++)
            ofs << obj[i][j] << "\t";
        ofs << "\n";
    }
    ofs.close();
}

// Calculates rotation matrix given euler angles.
cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta)
{
    // Calculate rotation about roll.
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
    
    // Calculate rotation about pitch.
    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
    
    // Calculate rotation about yaw.
    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
    
    
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;
    
    return R;

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

int Calibrate::save_chessboard(cv::VideoCapture &cap, int delay, std::string image_name, std::string ext) const
{
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
        if (img.channels() == 1)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

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

    write_to_file(object_filename, object_points);
    std::cout << "Saved object points to " << object_filename << std::endl;

    write_to_file(image_filename, image_points);
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
                                 cv::Size imageSize)
{
    std::vector<cv::Mat> rvecs, tvecs;
    std::cout << "Calibrating!" << std::endl;
    cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    std::cout << "Calibration done!" << std::endl;

    for (int i = 0; i < cameraMatrix.rows; i++)
    {
        for (int j = 0; j < cameraMatrix.cols; j++)
            std::cout << cameraMatrix.at<float>(i, j) << "\t";
        std::cout << "\n";
    }
    
   calculateImagePoints(imagePoints, objectPoints, rvecs, tvecs); 
   
}

void Calibrate::displayImagePoints(cv::Mat &img, const std::vector<cv::Point2f> &imagePoints)
{
    CHECK(!img.empty());
    for (size_t i = 0; i < imagePoints.size(); i++) 
        cv::circle(img, imagePoints[i], 5, cv::Scalar(0,0,255));
    
}

void Calibrate::calculateImagePoints(const std::vector<std::vector<cv::Point2f>> &imagePoints,
                            const std::vector<std::vector<cv::Point3f>> &objectPoints,
                            std::vector<cv::Mat> rvecs,
                            std::vector<cv::Mat> tvecs)
{
    cv::Mat RotationMatrix;
    cv::Rodrigues(rvecs[0], RotationMatrix);
    cv::Vec3f RPY(0, -CV_PI*30/180, 0);
    

    for(int i = 0; i < objectPoints.size(); i++)
    {   
        //rvecs[i] = 0; // if you want the see object coordinates.
        //tvecs[i] = 0;

        cv::Rodrigues(RotationMatrix*eulerAnglesToRotationMatrix(RPY), rvecs[i]); // same rotation for every picture.
        cv::projectPoints(objectPoints[0], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints[i]);
        
        //DEBUG(rvecs[i]);
        //DEBUG(tvecs[i]);
        
    }
}

void Calibrate::display_undistorted(cv::Mat &img, const std::vector<std::vector<cv::Point2f>> &imagePoints, std::string winname)
{
    CHECK(!img.empty());
    bool show_once = winname.empty();
    if (show_once)
        winname = "undistorted";
    cv::Mat imageUndistorted;
    undistort(img, imageUndistorted, cameraMatrix, distCoeffs);
    imshow(winname, imageUndistorted);

    char ch = 0;
    
    
    if (show_once)
    {
        char ch = cv::waitKey(0);
        CHECK(ch != 27);
        img = imageUndistorted.clone();
    }
}

void Calibrate::display_undistorted(cv::VideoCapture &cap, const std::vector<std::vector<cv::Point2f>> &imagePoints, std::string winname)
{
    // make sure image stream is open
    CHECK(cap.isOpened());
    cv::Mat img;
    cap >> img;
    CHECK(!img.empty());
    CHECK(!winname.empty());
    
    // for display purposes
    cv::namedWindow(winname);

    int i = 0;
    char ch = 0;
    while (ch != 27) // ESC is pressed
    {
        if (img.empty())
        {
            std::cout << "End of image stream!" << std::endl;
            break;
        }
        displayImagePoints(img, imagePoints[i]);
        DEBUG(imagePoints[i]);    
        imshow("objectPoints", img); 
        display_undistorted(img, imagePoints, winname);
        ch = cv::waitKey();
        cap >> img;
        i++;
    }
    cv::destroyWindow(winname);
}

