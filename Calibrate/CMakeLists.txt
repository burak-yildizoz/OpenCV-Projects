cmake_minimum_required(VERSION 2.8.3)
project(calibrate)

if(WIN32)
  # put opencv_world???.dll near the final executable before running
  # which is located at "${OpenCV_DIR}/../bin"
  set(OpenCV_DIR "C:/opencv/build/x64/vc15/lib")
endif(WIN32)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wall)
if(NOT WIN32)
  add_compile_options(-Wextra -Wpedantic)
  add_compile_options(-pthread)
endif(NOT WIN32)

include_directories("include")

set(SOURCES "src/Calibrate.cpp")

add_executable(pinhole main_calibrate.cpp ${SOURCES})
target_link_libraries(pinhole ${OpenCV_LIBS})

list(APPEND SOURCES "src/Fisheye.cpp")

add_executable(fisheye main_fisheye.cpp ${SOURCES})
target_link_libraries(fisheye ${OpenCV_LIBS})

