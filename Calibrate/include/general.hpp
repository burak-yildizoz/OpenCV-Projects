#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <csignal>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>

#define DEBUG(cc)                                                              \
  if (true) {                                                                  \
    std::cout << #cc << " = " << std::endl;                                    \
    std::cout << cc << std::endl;                                              \
  }

#define CHECK(cc)                                                              \
  if (!(cc)) {                                                                 \
    std::cerr << "Error! " << #cc << std::endl;                                \
    std::cerr << "Line: " << __LINE__ << std::endl;                            \
    std::cerr << "In file: " << __FILE__ << std::endl;                         \
    raise(SIGTERM);                                                            \
  }

// print vector container elements
template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  for (unsigned i = 0; i < v.size(); i++) {
    os << v[i];
    if (i != v.size() - 1)
      os << " ";
  }
  return os;
}

// print matrix container elements
template <class T>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<std::vector<T>> &M) {
  for (unsigned i = 0; i < M.size(); i++)
    os << M[i] << "\n";
  return os;
}

// read until *ch* comes
static void _read(std::ifstream &is, const char ch) {
  char _ch = 0;
  while (is.good() && ch != _ch)
    is >> _ch;
}

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

#endif // GENERAL_HPP
