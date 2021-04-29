#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/videoio.hpp>
#include <regex>
#include <stdexcept>
#include <vector>

#define DEBUG(cc)                                                              \
  if (true) {                                                                  \
    std::cout << #cc << " =\n" << cc << std::endl;                             \
  }

#define CHECK(cc)                                                              \
  if (!(cc)) {                                                                 \
    std::cerr << "Error: " << #cc << std::endl;                                \
    std::cerr << "Line : " << __LINE__ << std::endl;                           \
    std::cerr << "File : " << __FILE__ << std::endl;                           \
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');        \
    std::cin.get();                                                            \
    raise(SIGTERM);                                                            \
  }

// print vector container
template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  for (size_t i = 0; i < v.size(); i++)
    os << v[i] << " ";
  return os;
}

// print matrix container
template <class T>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<std::vector<T>> &M) {
  for (size_t i = 0; i < M.size(); i++)
    os << M[i] << "\n";
  os << std::endl;
  return os;
}

// print keypoint
inline std::ostream &operator<<(std::ostream &os, const cv::KeyPoint &kp) {
  os << kp.pt;
  return os;
}

// print DMatch
inline std::ostream &operator<<(std::ostream &os, const cv::DMatch &d) {
  os << "distance\t" << d.distance << "\n";
  os << "imgIdx\t" << d.imgIdx << "\n";
  os << "queryIdx\t" << d.queryIdx << "\n";
  os << "trainIdx\t" << d.trainIdx << "\n\n";
  return os;
}

namespace {
// read until *ch* comes
inline void _read(std::ifstream &is, const char ch) {
  char _ch = 0;
  while (is.good() && ch != _ch)
    is >> _ch;
}
} // namespace

// read Point3
template <typename _Tp>
std::ifstream &operator>>(std::ifstream &is, cv::Point3_<_Tp> &v) {
  _read(is, '[');
  is >> v.x;
  _read(is, ',');
  is >> v.y;
  _read(is, ',');
  is >> v.z;
  _read(is, ']');
  return is;
}

// read Point
template <typename _Tp>
std::ifstream &operator>>(std::ifstream &is, cv::Point_<_Tp> &v) {
  _read(is, '[');
  is >> v.x;
  _read(is, ',');
  is >> v.y;
  _read(is, ']');
  return is;
}

namespace general {
inline void matwrite(const std::string &filename, const cv::Mat &mat) {
  std::ofstream fs(filename, std::fstream::binary);

  // Header
  int type = mat.type();
  int channels = mat.channels();
  fs.write((char *)&mat.rows, sizeof(int)); // rows
  fs.write((char *)&mat.cols, sizeof(int)); // cols
  fs.write((char *)&type, sizeof(int));     // type
  fs.write((char *)&channels, sizeof(int)); // channels

  // Data
  if (mat.isContinuous()) {
    fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
  } else {
    int rowsz = CV_ELEM_SIZE(type) * mat.cols;
    for (int r = 0; r < mat.rows; ++r) {
      fs.write(mat.ptr<char>(r), rowsz);
    }
  }
}

inline cv::Mat matread(const std::string &filename) {
  std::ifstream fs(filename, std::fstream::binary);

  // Header
  int rows, cols, type, channels;
  fs.read((char *)&rows, sizeof(int));     // rows
  fs.read((char *)&cols, sizeof(int));     // cols
  fs.read((char *)&type, sizeof(int));     // type
  fs.read((char *)&channels, sizeof(int)); // channels

  // Data
  cv::Mat mat(rows, cols, type);
  fs.read((char *)mat.data, CV_ELEM_SIZE(type) * rows * cols);

  return mat;
}

// find the most frequent element in any container
// {mostFrequentElement, maxFrequency}
template <typename T>
std::pair<typename T::value_type, int> most_frequent_element(T const &v) {
  // Precondition: v is not empty
  std::map<typename T::value_type, int> frequencyMap;
  int maxFrequency = 0;
  typename T::value_type mostFrequentElement{};
  for (auto &&x : v) {
    int f = ++frequencyMap[x];
    if (f > maxFrequency) {
      maxFrequency = f;
      mostFrequentElement = x;
    }
  }
  return std::make_pair(mostFrequentElement, maxFrequency);
}

inline bool videocapture_open(cv::VideoCapture &cap, const std::string &input) {
  std::regex img_seq(R"(%\d*d)"); // e.g. "%d" or "%01d"
  std::regex url(R"(:\/\/)");     // "://"
  if (input.size() == 1)
    cap.open(atoi(input.c_str()));
  else if (std::regex_search(input, img_seq) || std::regex_search(input, url))
    cap.open(input);
  else
    cap.open(cv::samples::findFile(input));
  return cap.isOpened();
}

// format string like printf
// https://stackoverflow.com/a/26221725/12447766
template <typename... Args>
std::string string_format(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

inline std::string &toupper(std::string &s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}
} // namespace general

#endif // GENERAL_HPP
