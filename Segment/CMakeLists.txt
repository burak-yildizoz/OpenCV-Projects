cmake_minimum_required(VERSION 2.8.3)
project(segment)

if(WIN32)
  # put opencv_world???.dll near the final executable before running
  # which is located at "${OpenCV_DIR}/../bin"
  # put egbis.dll too
  set(OpenCV_DIR "C:/opencv/build/x64/vc15/lib")
endif(WIN32)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
if(NOT WIN32)
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-pthread)
endif(NOT WIN32)

include_directories("include")
include_directories("egbis")
include_directories("../Stitch/include")

if(WIN32)
  set(EGBIS_LIB "${CMAKE_SOURCE_DIR}/egbis/build/Release/egbis.lib")
  set(STITCH_LIB "${CMAKE_SOURCE_DIR}/../Stitch/build/Release/stitch.lib")
else()
  set(EGBIS_LIB "${CMAKE_SOURCE_DIR}/egbis/build/libegbis.so")
  set(STITCH_LIB "${CMAKE_SOURCE_DIR}/../Stitch/build/libstitch.so")
endif(WIN32)

add_executable(main main.cpp src/Segmenter.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${EGBIS_LIB} ${STITCH_LIB})
