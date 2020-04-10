#include <insightface.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage: ./" << argv[0] << " model_path device image" << std::endl;
    return -1;
  }
  std::string model_path = argv[1];
  std::string device = argv[2];
  std::string image_name = argv[3];
  InsightFace face(model_path, device);
  if (face.init() < 0) {
    std::cout << "InsightFace init fail" << std::endl;
    return -2;
  }
  cv::Mat image = cv::imread(image_name);
  if (image.empty()) {
    std::cout << "Invalid image" << std::endl;
    return -3;
  }
  std::vector<float> feature;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  face.process(image, feature);
  gettimeofday(&end, NULL);
  for (int i = 0; i < feature.size(); ++i) {
    std::cout << feature[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Elapsed Time: " << (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec) / 1000 << " ms" << std::endl;
  return 0;
}
