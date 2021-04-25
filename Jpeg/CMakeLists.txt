cmake_minimum_required(VERSION 2.8.3)
project(jpeg_compression)

include(${CMAKE_CURRENT_LIST_DIR}/../CMake_OpenCV.txt)

add_executable(jpeg_compression jpeg_compression.cpp)
target_link_libraries(jpeg_compression ${OpenCV_LIBS})
