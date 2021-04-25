cmake_minimum_required(VERSION 2.8.3)
project(segment)

include(${CMAKE_CURRENT_LIST_DIR}/../CMake_OpenCV.txt)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/egbis)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../Stitch ${CMAKE_CURRENT_BINARY_DIR}/Stitch)

include_directories("include")

add_executable(segment main.cpp src/Segmenter.cpp)
target_link_libraries(segment ${OpenCV_LIBS} egbis_library stitch_library)
