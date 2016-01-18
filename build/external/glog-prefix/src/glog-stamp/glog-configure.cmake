

set(command "env;CFLAGS=-fPIC;CXXFLAGS=-fPIC;/home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog/configure;--prefix=/home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-install;--enable-shared=no;--enable-static=yes;--with-gflags=/home/zhaoyin/workspace/cifar/cifar_test/build/external/gflags-install/lib/..")
execute_process(
  COMMAND ${command}
  RESULT_VARIABLE result
  OUTPUT_FILE "/home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-configure-out.log"
  ERROR_FILE "/home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-configure-err.log"
  )
if(result)
  set(msg "Command failed: ${result}\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  set(msg "${msg}\nSee also\n  /home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-configure-*.log\n")
  message(FATAL_ERROR "${msg}")
else()
  set(msg "glog configure command succeeded.  See also /home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-configure-*.log\n")
  message(STATUS "${msg}")
endif()
