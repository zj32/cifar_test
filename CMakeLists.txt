cmake_minimum_required (VERSION 2.6)
project (cifar_test)
set(CMAKE_BUILD_TYPE Release)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

include(ExternalProject)
#add_subdirectory(src)

add_library(cifar_util_lib "src/cifar_util.cpp")

set(cifar_LINKER_LIBS "cifar_util_lib")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ Google-glog
include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND cifar_LINKER_LIBS ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND cifar_LINKER_LIBS ${GFLAGS_LIBRARIES})

# Threads
find_package(Threads)
list(APPEND cifar_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

# OpenCV
find_package(OpenCV REQUIRED)
list(APPEND cifar_LINKER_LIBS ${OpenCV_LIBS})

# Caffe
# include("cmake/External/CAFFEConfig.cmake")
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

#find_package(CAFFE REQUIRED)
#include_directories(${CAFFE_INCLUDE_DIRS})
list(APPEND cifar_LINKER_LIBS ${Caffe_LIBRARIES})

message(STATUS  "Libs include: ${cifar_LINKER_LIBS}")
message(STATUS "Currnet source dir =  ${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(cifar_test_main "src/cifar_test.cpp")
target_include_directories (cifar_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cifar_test_main ${cifar_LINKER_LIBS} )