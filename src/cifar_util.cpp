#include "cifar_util.h"

#include <fstream>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "glog/logging.h"


namespace cifar_util {
namespace {
constexpr int kCIFARImageNBytes = 3072;
constexpr int kCIFARImageStep = kCIFARImageNBytes / 3;
constexpr int kCIFARSize = 32;

CIFARImage ReadSingleImage(std::ifstream* file) {
	char label_char;
	file->read(&label_char, 1);
	char buffer[kCIFARImageNBytes];
    file->read(buffer, kCIFARImageNBytes);
	return CIFARImage(buffer, label_char);
}
}  // namespace

CIFARImage::CIFARImage(char* buffer, int label)
    : label_(label) {
    image_ = Mat(kCIFARSize, kCIFARSize, CV_8UC3);
    for (int i = 0; i < kCIFARImageStep; i++) {
    	const Vec3b color{
    		static_cast<unsigned char>(buffer[i + 2 * kCIFARImageStep]),
    		static_cast<unsigned char>(buffer[i + kCIFARImageStep]),
			static_cast<unsigned char>(buffer[i])};
    	image_.at<Vec3b>(i) = color;
    }
}

bool ConvertCifarDataSetToImages(const string& input_filename,
                                 const string& output_folder,
								 int num_image_to_read) {
    LOG(INFO) << "Input file = " << input_filename;
    std::ifstream data_file(
    		input_filename.c_str(), std::ios::in | std::ios::binary);

    CHECK(data_file) << "Unable to open train file #" << input_filename;

    for (int i = 0; i < num_image_to_read; i++) {
    	const CIFARImage image = ReadSingleImage(&data_file);
    	imshow("image", image.image());
    	LOG(INFO) << "Image index " << i << ", label = " << image.label();
    	waitKey(0);
    }
    LOG(INFO) << "Output folder = " << output_folder;
    return true;
}

}  // namespace cifar

