// g++ jpeg_compression.cpp -o jpeg_compression.exe -Wall -Wextra -Wpedantic -O2
// -std=c++11 $(pkg-config --libs --cflags opencv)

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define DEBUG(cc)                                                              \
  if (true) {                                                                  \
    std::cout << #cc << " = \n" << cc << "\n\n";                               \
  }

/**
 * return the smallest power of two value
 * greater than x
 *
 * Input range:  [2..2147483648]
 * Output range: [2..2147483648]
 *
 */
#ifdef _WIN32
static inline uint32_t pow2(uint32_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}
#else
__attribute__((const)) static inline uint32_t pow2(uint32_t x) {
  return 1 << (32 - __builtin_clz(x));
}
#endif

std::vector<std::pair<int, int>> zigzag_order() {
  // zigzag order (0,0) (0,1) (1,0) (2,0) (1,1) ...
  std::vector<std::pair<int, int>> res; // rvec, cvec
  int r = 0, c = 0;
  res.push_back(std::make_pair(r, c));
  for (int k = 1; k <= 7; k++) {
    if (k % 2) // right
      res.push_back(std::make_pair(r, ++c));
    else // down
      res.push_back(std::make_pair(++r, c));
    for (int q = 0; q < k; q++) {
      if (k % 2) // left down
        res.push_back(std::make_pair(++r, --c));
      else // right up
        res.push_back(std::make_pair(--r, ++c));
    }
  }
  res.push_back(std::make_pair(r, ++c));
  for (int k = 6; k >= 1; k--) {
    for (int q = 0; q < k; q++) {
      if (k % 2) // left down
        res.push_back(std::make_pair(++r, --c));
      else // right up
        res.push_back(std::make_pair(--r, ++c));
    }
    if (k % 2) // right
      res.push_back(std::make_pair(r, ++c));
    else // down
      res.push_back(std::make_pair(++r, c));
  }
  return res;
}

size_t outputSize(const cv::Mat &B) {
  assert(B.type() == CV_16S);
  assert(B.rows % 8 == 0);
  assert(B.cols % 8 == 0);
  double output_size = 0;
  std::vector<std::pair<int, int>> zigzag = zigzag_order();
  // slice zigzag order each 8x8 block
  int16_t dc = 0;
  for (int i = 0; i < B.rows; i += 8) {
    for (int j = 0; j < B.cols; j += 8) {
      cv::Mat image(B, cv::Rect(j, i, 8, 8));
      std::multiset<std::pair<int, int>> mset; // runlength, size
      int runlength = 0;
      int diff =
          static_cast<int>(sqrt(pow2(abs(image.at<int16_t>(0, 0) - dc))));
      mset.insert(std::pair<int, int>(runlength, diff));
      dc = image.at<int16_t>(0, 0);
      for (int k = 1; k < 64; k++) {
        int16_t &p = image.at<int16_t>(zigzag[k].first, zigzag[k].second);
        if (p == 0) {
          runlength++;
        } else {
          int sz = static_cast<int>(sqrt(pow2(abs(p))));
          while (runlength > 15) {
            mset.insert(std::pair<int, int>(15, 0));
            runlength -= 15;
          }
          mset.insert(std::pair<int, int>(runlength, sz));
          runlength = 0;
        }
      }
      mset.insert(std::pair<int, int>(0, 0));
      typedef std::multiset<std::pair<int, int>>::iterator It;
      It it = mset.begin();
      std::vector<std::pair<int, int>>
          probabilities; // occurance, index of mset
      while (it != mset.end()) {
        std::pair<It, It> ret = mset.equal_range(*it);
        probabilities.push_back(std::pair<int, int>(
            static_cast<int>(std::distance(ret.first, ret.second)),
            static_cast<int>(std::distance(mset.begin(), it))));
        it = ret.second;
      }
      std::vector<int> prob_idx(
          probabilities.size(),
          0); // huffman bit count of index of probabilities
      std::vector<std::pair<int, std::vector<int>>>
          tree; // occurance, indexes of probabilities
      for (unsigned k = 0; k < probabilities.size(); k++) {
        tree.push_back(
            std::make_pair(probabilities[k].first, std::vector<int>(1, k)));
      }
      while (tree.size() != 1) {
        std::sort(tree.begin(), tree.end(),
                  [](const std::pair<int, std::vector<int>> &lhs,
                     const std::pair<int, std::vector<int>> &rhs) {
                    return lhs.first < rhs.first;
                  });
        tree[1].first += tree[0].first; // occurance
        for (unsigned k = 0; k < tree[0].second.size(); k++)
          tree[1].second.push_back(tree[0].second[k]); // index of probabilities
        for (unsigned k = 0; k < tree[1].second.size(); k++)
          prob_idx[k]++; // huffman bit count
        tree.erase(tree.begin());
      }
      for (unsigned k = 0; k < probabilities.size(); k++) {
        int occurance = probabilities[k].first;
        It it = mset.begin();
        std::advance(it, probabilities[k].second);
        int sz = it->second;
        int huff_count = prob_idx[k];
        output_size += occurance * (huff_count + sz);
      }
    }
  }
  return static_cast<size_t>(output_size);
}

cv::Mat quantize(unsigned quality, bool chrominance) {
  cv::Mat Tb;
  if (!chrominance) {
    double T[8][8] = {{16, 11, 10, 16, 24, 40, 51, 61},
                      {12, 12, 14, 19, 26, 58, 60, 55},
                      {14, 13, 16, 24, 40, 57, 69, 56},
                      {14, 17, 22, 29, 51, 87, 80, 62},
                      {18, 22, 37, 56, 68, 109, 103, 77},
                      {24, 35, 55, 64, 81, 104, 113, 92},
                      {49, 64, 78, 87, 103, 121, 120, 101},
                      {72, 92, 95, 98, 112, 100, 103, 99}};
    Tb = cv::Mat(8, 8, CV_64F, T).clone();
  } else {
    double T[8][8] = {
        {17, 18, 24, 47, 99, 99, 99, 99}, {18, 21, 26, 66, 99, 99, 99, 99},
        {24, 26, 56, 99, 99, 99, 99, 99}, {47, 66, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99}, {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99}, {99, 99, 99, 99, 99, 99, 99, 99}};
    Tb = cv::Mat(8, 8, CV_64F, T).clone();
  }
  double S = (quality < 50) ? (5000.0 / quality) : (200 - 2 * quality);
  cv::Mat Ts = (S * Tb) / 100;
  Ts.convertTo(Ts, CV_16S);
  Ts.setTo(1, Ts == 0);
  return Ts;
}

cv::Mat encode(const cv::Mat &A, unsigned quality, bool chrominance = false) {
  if (quality == 0)
    quality = 1;
  assert(quality <= 100);
  assert(A.type() == CV_8U);
  const int rows = A.rows + (7 - (((A.rows % 8) + 7) % 8));
  const int cols = A.cols + (7 - (((A.cols % 8) + 7) % 8));
  cv::Mat g = cv::Mat::zeros(rows, cols, A.type());
  A.copyTo(g(cv::Rect(0, 0, A.cols, A.rows)));
  g.convertTo(g, CV_64F);
  g = g - 128;
  cv::Mat G = cv::Mat::zeros(rows, cols, CV_64F);
  for (int i = 0; i < rows; i += 8) {
    for (int j = 0; j < cols; j += 8) {
      cv::Rect rect(j, i, 8, 8);
      cv::dct(cv::Mat(g, rect), cv::Mat(G, rect));
    }
  }
  cv::Mat Q = repeat(quantize(quality, chrominance), rows / 8, cols / 8);
  Q.convertTo(Q, CV_64F);
  cv::Mat B;
  cv::divide(G, Q, B);
  B.convertTo(B, CV_16S);
  return B;
}

cv::Mat decode(const cv::Mat &B, unsigned quality, bool chrominance = false) {
  if (quality == 0)
    quality = 1;
  assert(quality <= 100);
  assert(B.type() == CV_16S);
  assert(B.rows % 8 == 0);
  assert(B.cols % 8 == 0);
  const int rows = B.rows;
  const int cols = B.cols;
  cv::Mat Q = repeat(quantize(quality, chrominance), rows / 8, cols / 8);
  Q.convertTo(Q, CV_64F);
  cv::Mat BB = cv::Mat::zeros(rows, cols, B.type());
  B.copyTo(BB(cv::Rect(0, 0, B.cols, B.rows)));
  cv::Mat F;
  cv::multiply(BB, Q, F, 1, CV_64F);
  cv::Mat f = cv::Mat::zeros(rows, cols, CV_64F);
  for (int i = 0; i < rows; i += 8) {
    for (int j = 0; j < cols; j += 8) {
      cv::Rect rect(j, i, 8, 8);
      cv::idct(cv::Mat(F, rect), cv::Mat(f, rect));
    }
  }
  cv::Mat A = f + 128;
  A.convertTo(A, CV_8U);
  return A;
}

void trackbar_callback(int quality, void *image_p) {
  if (quality == 0)
    quality = 1;
  std::pair<const cv::Mat &, cv::Mat &> *image =
      (std::pair<const cv::Mat &, cv::Mat &> *)image_p;
  const cv::Mat &A = image->first;
  cv::Mat &disp_image = image->second;
  size_t output_size = 0;
  if (A.type() == CV_8UC1) {
    cv::Mat B = encode(A, quality);
    output_size += outputSize(B);
    cv::Mat Ad = decode(B, quality);
    Ad = Ad(cv::Rect(0, 0, A.cols, A.rows));
    // cv::resize(Ad, disp_image, cv::Size(400, 400 * Ad.rows / Ad.cols), 0, 0,
    // cv::INTER_AREA);
  } else if (A.type() == CV_8UC3) {
    cv::Mat YUV;
    cvtColor(A, YUV, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> yuv_vec;
    cv::split(YUV, yuv_vec);
    cv::Mat Y_en = encode(yuv_vec[0], quality);
    cv::Mat Cr_en = encode(yuv_vec[1], quality, true);
    cv::Mat Cb_en = encode(yuv_vec[2], quality, true);
    output_size += outputSize(Y_en) + outputSize(Cr_en) + outputSize(Cb_en);
    cv::Mat Y = decode(Y_en, quality);
    cv::Mat Cr = decode(Cr_en, quality, true);
    cv::Mat Cb = decode(Cb_en, quality, true);
    yuv_vec[0] = Y;
    yuv_vec[1] = Cr;
    yuv_vec[2] = Cb;
    cv::merge(yuv_vec, YUV);
    cvtColor(YUV, disp_image, cv::COLOR_YCrCb2BGR);
    disp_image = disp_image(cv::Rect(0, 0, A.cols, A.rows));
    // cv::resize(disp_image, disp_image, cv::Size(400, 400 * disp_image.rows /
    // disp_image.cols), 0, 0, cv::INTER_AREA);
  } else
    assert((A.type() == CV_8UC1) || (A.type() == CV_8UC3));
  double ratio = (double)output_size / (A.rows * A.cols * A.channels());
  printf("\rCompression ratio: %.2f%%  ", 100 * (1 - ratio));
  fflush(stdout);
}

void show_quantization(const cv::Mat &A) {
  int quality = 50;
  const char winname[] = "Quantized";
  cv::namedWindow(winname);
  cv::Mat disp_image;
  std::pair<const cv::Mat &, cv::Mat &> image = {A, disp_image};
  cv::createTrackbar("quality", winname, &quality, 100, trackbar_callback,
                     (void *)&image);
  trackbar_callback(quality, (void *)&image);
  char ch = 0;
  while (ch != 27) // Press ESC to quit
  {
    cv::imshow(winname, disp_image);
    ch = cv::waitKey(1);
    if (ch == ' ') {
      std::string filename = "A_" + std::to_string(quality) + ".png";
      std::cout << "Saving image" << filename << std::endl;
      cv::imwrite(filename, disp_image);
    }
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  std::string input = "armut.bmp";
  if (argc == 2)
    input = argv[1];
  DEBUG(input);
  cv::Mat A = cv::imread(input);
  assert(!A.empty());
  cv::Mat disp_orig = A.clone();
  // cv::resize(disp_orig, disp_orig, cv::Size(400, 400 * A.rows / A.cols), 0,
  // 0, cv::INTER_AREA);
  cv::imshow("Original", disp_orig);
  show_quantization(A);
  return 0;
}
