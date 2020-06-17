#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <iostream>
#include <csignal>
#include <vector>
#include <opencv2/core.hpp>

#define DEBUG(cc) if(true) \
{ \
    std::cout << #cc << " =\n" << cc << std::endl; \
}

#define CHECK(cc) if(!(cc)) \
{ \
    std::cerr << "Error: " << #cc << std::endl; \
    std::cerr << "Line : " << __LINE__ << std::endl; \
    std::cerr << "File : " << __FILE__ << std::endl; \
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); \
    std::cin.get(); \
    raise(SIGTERM); \
}

// print vector container
template <class T>
inline std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
    for (size_t i=0; i<v.size(); i++)
        os << v[i] << " ";
    return os;
}

// print matrix container
template <class T>
inline std::ostream& operator << (std::ostream& os, const std::vector< std::vector<T> >& M)
{
    for (size_t i=0; i<M.size(); i++)
        os << M[i] << "\n";
    os << std::endl;
    return os;
}

// print keypoint
inline std::ostream& operator << (std::ostream& os, const cv::KeyPoint& kp)
{
    os << kp.pt;
    return os;
}

// print DMatch
inline std::ostream& operator << (std::ostream& os, const cv::DMatch& d)
{
    os << "distance\t" << d.distance << "\n";
    os << "imgIdx\t" << d.imgIdx << "\n";
    os << "queryIdx\t" << d.queryIdx << "\n";
    os << "trainIdx\t" << d.trainIdx << "\n\n";
    return os;
}

// read data to cv::Mat of type CV_64F
inline std::istream& operator >> (std::istream& is, cv::Mat& mat)
{
    char ch = 0;
    while (is.good() && ch != '[')
        is >> ch;
    CHECK(ch == '[');
    std::vector<double> m(0);
    size_t cols = 0;
    bool end_of_file = false;
    while ( is.good() && ! end_of_file )
    {
        bool end_of_line = false;
        std::vector<double> v(0);
        while ( is.good() && ! end_of_line )
        {
            double d;
            is >> d;
            v.push_back(d);
            char c;
            is >> c;
            switch (c)
            {
                case ']':
                end_of_file = true;
                case ';':
                end_of_line = true;
                break;
                case ',':
                break;
                default:
                std::cerr << "Unexpected character read: " << c << std::endl;
                CHECK(false);
            }
        }
        if (m.size() == 0)
            cols = v.size();
        else
            CHECK(v.size() == cols);
        m.insert(m.end(), v.begin(), v.end());
    }
    cv::Mat res(cv::Size(cols, m.size() / cols), CV_64F, m.data());
    mat = res.clone();
    return is;
}

#endif // GENERAL_HPP

