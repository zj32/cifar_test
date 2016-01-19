#include "cifar_util.h"

#include "caffe/caffe.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"

DEFINE_string(input_filename,
        "/home/zhaoyin/Documents/caffe-master/data/cifar10/test_batch.bin",
        "Input filename.");
DEFINE_int32(num_images, 100, "Number of images to process");

using namespace std;
using caffe::Net;

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr=true;
  CHECK(!FLAGS_input_filename.empty());
  const vector<cifar_util::CIFARImage> images =
      cifar_util::ConvertCifarDataSetToImages(
          FLAGS_input_filename, FLAGS_num_images);
  LOG(INFO) << "Finished reading " << images.size();

  // TODO: load a pretrained net and test on the testing images.
  // Net<float> *caffe_net;
  // caffe_net->ClearParamDiffs();
	return 0;
}
