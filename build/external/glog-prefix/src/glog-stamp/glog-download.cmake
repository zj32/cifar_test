

set(command "/usr/bin/cmake;-P;/home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/tmp/glog-gitclone.cmake")
execute_process(
  COMMAND ${command}
  RESULT_VARIABLE result
  OUTPUT_FILE "/home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-download-out.log"
  ERROR_FILE "/home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-download-err.log"
  )
if(result)
  set(msg "Command failed: ${result}\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  set(msg "${msg}\nSee also\n  /home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-download-*.log\n")
  message(FATAL_ERROR "${msg}")
else()
  set(msg "glog download command succeeded.  See also /home/zhaoyin/workspace/cifar/cifar_test/build/external/glog-prefix/src/glog-stamp/glog-download-*.log\n")
  message(STATUS "${msg}")
endif()
