cmake_minimum_required(VERSION 2.8.3)
project(jpeg_compression)

if(NOT TARGET general_library)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/general EXCLUDE_FROM_ALL)
endif(NOT TARGET general_library)
find_opencv()

add_executable(jpeg_compression jpeg_compression.cpp)
target_link_libraries(jpeg_compression ${OpenCV_LIBS})
