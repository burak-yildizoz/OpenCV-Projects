#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <iostream>
#include <csignal>
#include <vector>
#include <fstream>
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

inline void matwrite(const std::string& filename, const cv::Mat& mat)
{
    std::ofstream fs(filename, std::fstream::binary);

    // Header
    int type = mat.type();
    int channels = mat.channels();
    fs.write((char*)&mat.rows, sizeof(int));    // rows
    fs.write((char*)&mat.cols, sizeof(int));    // cols
    fs.write((char*)&type, sizeof(int));        // type
    fs.write((char*)&channels, sizeof(int));    // channels

    // Data
    if (mat.isContinuous())
    {
        fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    }
    else
    {
        int rowsz = CV_ELEM_SIZE(type) * mat.cols;
        for (int r = 0; r < mat.rows; ++r)
        {
            fs.write(mat.ptr<char>(r), rowsz);
        }
    }
}

inline cv::Mat matread(const std::string& filename)
{
    std::ifstream fs(filename, std::fstream::binary);

    // Header
    int rows, cols, type, channels;
    fs.read((char*)&rows, sizeof(int));         // rows
    fs.read((char*)&cols, sizeof(int));         // cols
    fs.read((char*)&type, sizeof(int));         // type
    fs.read((char*)&channels, sizeof(int));     // channels

    // Data
    cv::Mat mat(rows, cols, type);
    fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

    return mat;
}

// https://stackoverflow.com/questions/8936063/does-there-exist-a-static-warning#answer-8990275
#if defined(__GNUC__)
#define DEPRECATE(foo, msg) foo __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define DEPRECATE(foo, msg) __declspec(deprecated(msg)) foo
#else
#error This compiler is not supported
#endif

#define PP_CAT(x,y) PP_CAT1(x,y)
#define PP_CAT1(x,y) x##y

namespace detail
{
    struct true_type {};
    struct false_type {};
    template <int test> struct converter : public true_type {};
    template <> struct converter<0> : public false_type {};
}

#define STATIC_WARNING(cond, msg) \
struct PP_CAT(static_warning,__LINE__) { \
  DEPRECATE(void _(::detail::false_type const& ),msg) {}; \
  void _(::detail::true_type const& ) {}; \
  PP_CAT(static_warning,__LINE__)() {_(::detail::converter<(cond)>());} \
}

#endif // GENERAL_HPP
