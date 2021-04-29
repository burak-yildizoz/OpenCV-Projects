cmake_minimum_required(VERSION 2.8.3)
project(segment)

if(NOT TARGET general_library)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/general EXCLUDE_FROM_ALL)
endif(NOT TARGET general_library)
if(NOT TARGET egbis_library)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/egbis EXCLUDE_FROM_ALL)
endif(NOT TARGET egbis_library)
if(NOT TARGET stitch_library)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../Stitch ${CMAKE_CURRENT_BINARY_DIR}/Stitch EXCLUDE_FROM_ALL)
endif(NOT TARGET stitch_library)
find_opencv()

include_directories("include")

add_executable(segment main.cpp src/Segmenter.cpp)
target_link_libraries(segment egbis_library stitch_library)
