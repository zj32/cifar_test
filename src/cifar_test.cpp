#include "cifar_util.h"

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"

DEFINE_string(input_filename,
        "/home/zhaoyin/Documents/caffe-master/data/cifar10/test_batch.bin",
        "Input filename.");
DEFINE_int32(num_images, 100, "Number of images to process");

using namespace std;
using namespace cv;
using namespace caffe;

namespace {

int ClassifySingleImage(const cifar_util::CIFARImage& cifar_image,
                         Net<float>* caffe_net) {
  const Mat& image = cifar_image.image();
  std::vector<cv::Mat> images(1, image);
  std::vector<int> labels(1, 0);

  boost::shared_ptr<MemoryDataLayer<float>> md_layer =
      boost::dynamic_pointer_cast <MemoryDataLayer<float>>(
          caffe_net->layers()[0]);
  if (!md_layer) {
    LOG(ERROR) << "The first layer is not a MemoryDataLayer!\n";
    return -1;
  }

  md_layer->AddMatVector(images, labels);
  float loss;

  caffe_net->ForwardPrefilled(&loss);
  std::cout << "loss: " << loss << "\n";
  // read the 'ip2' layer and get the result
  boost::shared_ptr<Blob<float>> prob = caffe_net->blob_by_name("ip2");
  float maxval= 0;
  int   maxinx= 0;
  for (int i = 0; i < prob->count(); i++) {
    float val= prob->cpu_data()[i];
    if (val> maxval) {
        maxval= val;
        maxinx= i;
    }
    std::cout << "[" << i << "]" << val<< "\n";
  }
  std::cout << "Max value = " << maxval<< ", Max index = " << maxinx<<
      ", Ground truth = " << cifar_image.label() << "\n";
  return 0;
}
}  // namespace

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr=true;
  CHECK(!FLAGS_input_filename.empty());
  const vector<cifar_util::CIFARImage> cifar_images =
      cifar_util::ConvertCifarDataSetToImages(
          FLAGS_input_filename, FLAGS_num_images);
  LOG(INFO) << "Finished reading " << cifar_images.size();

  // TODO: load a pretrained net and test on the testing images.

  // fragment of C++ code:
  constexpr char kNetModelName[] =
      "/home/zhaoyin/workspace/cifar/cifar_test/"
      "cifar10_quick.prototxt";
  constexpr char kTrainedFilaName[] =
      "/home/zhaoyin/workspace/cifar/cifar_test/"
      "cifar10_quick_iter_5000.caffemodel.h5";

  Net<float> caffe_net(kNetModelName, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(kTrainedFilaName);
  LOG(INFO) << "Model is successfully loaded";
  Timer timer;
  timer.Start();
  for (const auto& cifar_image : cifar_images) {
    CHECK_GE(ClassifySingleImage(cifar_image, &caffe_net), 0);
  }
  timer.Stop();
  printf("Classifier %zu images used %.2f ms.\n", cifar_images.size(),
         timer.MilliSeconds());

	return 0;
}
