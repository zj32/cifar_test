#include "cifar_util.h"

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "glog/logging.h"
#include "gflags/gflags.h"

DEFINE_string(input_filename,
        "/home/zhaoyin/Documents/caffe-master/data/cifar10/test_batch.bin",
        "Input filename.");
DEFINE_int32(num_images, 100, "Number of images to process");

DEFINE_bool(use_gpu, true, "If true, use GPU.");

using namespace caffe;
using namespace cv;
using namespace std;

namespace {

bool ClassifySingleImage(const cifar_util::CIFARImage& cifar_image,
                         Net<float>* caffe_net) {
  const Mat& image = cifar_image.image();
  std::vector<cv::Mat> images(1, image);
  std::vector<int> labels(1, 0);

  boost::shared_ptr<MemoryDataLayer<float>> md_layer =
      boost::dynamic_pointer_cast <MemoryDataLayer<float>>(
          caffe_net->layers()[0]);
  CHECK(md_layer != nullptr) << "The first layer is not a MemoryDataLayer!\n";

  md_layer->AddMatVector(images, labels);
  float loss;

  caffe_net->ForwardPrefilled(&loss);
  std::cout << "loss: " << loss << "\n";
  // Read the 'ip2' layer and get the result.
  boost::shared_ptr<Blob<float>> prob = caffe_net->blob_by_name("ip2");
  float max_value = 0.0;
  int max_index = 0;
  for (int i = 0; i < prob->count(); i++) {
    float val= prob->cpu_data()[i];
    if (val > max_value) {
      max_value= val;
      max_index= i;
    }
    std::cout << "[" << i << "]" << val<< ", ";
  }
  const bool is_correct = (cifar_image.label() == max_index);
  std::cout << "Max value = " << max_value << ", Max index = " << max_index
      << ", Ground truth = " << cifar_image.label()
      << (is_correct ? ", Correct." : ", Wrong.") << "\n";
  return is_correct;
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

  // Absolute path of the model files.
  constexpr char kNetModelName[] =
      "../cifar10_quick.prototxt";
  constexpr char kTrainedFilaName[] =
      "../cifar10_quick_iter_5000.caffemodel.h5";

  if (FLAGS_use_gpu) {
    constexpr int kGPUDeviceId = 0;
    Caffe::SetDevice(kGPUDeviceId);
    Caffe::set_mode(Caffe::GPU);
  } else {
    Caffe::set_mode(Caffe::CPU);
  }

  Net<float> caffe_net(kNetModelName, caffe::TEST);

  caffe_net.CopyTrainedLayersFrom(kTrainedFilaName);
  LOG(INFO) << "Model is successfully loaded";
  Timer timer;
  timer.Start();
  int num_correct = 0;
  for (const auto& cifar_image : cifar_images) {
    num_correct +=
        static_cast<int>(ClassifySingleImage(cifar_image, &caffe_net));
  }
  timer.Stop();
  const double acc = static_cast<double>(num_correct) /
      static_cast<double>(cifar_images.size());
  printf("Classifier %zu images used %.2f ms. Acc = %.2f\n",
         cifar_images.size(), timer.MilliSeconds(), acc);
	return 0;
}
