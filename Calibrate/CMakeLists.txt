cmake_minimum_required(VERSION 2.8.3)
project(calibrate)

if(NOT TARGET general_library)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/general EXCLUDE_FROM_ALL)
endif(NOT TARGET general_library)
find_opencv()

include_directories("include")

set(SOURCES "src/Calibrate.cpp" "src/Fisheye.cpp")

add_executable(calibrate main_calibrate.cpp ${SOURCES})
target_link_libraries(calibrate general_library)
