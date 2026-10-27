cmake_minimum_required(VERSION 3.11)
project(im_store CXX)

find_package(pxl REQUIRED)
add_library(im_store SHARED im_store.cpp)
target_compile_features(im_store PRIVATE cxx_std_17)
target_link_libraries(im_store PRIVATE pxl::pxl)
