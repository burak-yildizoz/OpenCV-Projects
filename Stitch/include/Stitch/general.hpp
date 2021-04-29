#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
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
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  for (size_t i = 0; i < v.size(); i++)
    os << v[i] << " ";
  return os;
}

// print matrix container
template <class T>
inline std::ostream &operator<<(std::ostream &os,
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
} // namespace general

#endif // GENERAL_HPP
