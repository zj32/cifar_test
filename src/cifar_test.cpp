#include "cifar_util.h"

#include "glog/logging.h"
#include "gflags/gflags.h"

DEFINE_string(input_filename, "", "Input filename");

using namespace std;

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_alsologtostderr=true;
    CHECK(!FLAGS_input_filename.empty());
    cifar_util::ConvertCifarDataSetToImages(
    		FLAGS_input_filename, "output", 100);
	return 0;
}
