#ifndef CIFAR_UTIL_H_
#define CIFAR_UTIL_H_

#include <string>

#include "opencv/cv.h"
#include "opencv/highgui.h"

namespace cifar_util {
using namespace std;
using namespace cv;

class CIFARImage {
  public:
	CIFARImage(char* buffer, int label);
	const Mat& image() const { return image_; }
	int label() const { return label_; }
	// TODO: convert to caffe Datum

  private:
	Mat image_;
	const int label_;
};

vector<CIFARImage> ConvertCifarDataSetToImages(
        const string& input_file,
        int num_image_to_read);



}  // namespace cifar_util

#endif /* CIFAR_UTIL_H_ */
