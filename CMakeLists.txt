project(waifu2x-ncnn-vulkan)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)

# change these path to yours
include_directories("/home/nihui/osd/ncnn/build/install/include")
link_directories("/home/nihui/osd/ncnn/build/install/lib")

add_executable(waifu2x waifu2x.cpp)

target_link_libraries(waifu2x ncnn ${Vulkan_LIBRARY} ${OpenMP_CXX_LIBRARIES})
