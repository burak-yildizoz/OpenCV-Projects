cmake_minimum_required(VERSION 2.8.3)
project(calibrate)

include(${CMAKE_CURRENT_LIST_DIR}/../CMake_OpenCV.txt)

include_directories("include")

set(SOURCES "src/Calibrate.cpp")

add_executable(pinhole main_calibrate.cpp ${SOURCES})
target_link_libraries(pinhole ${OpenCV_LIBS})

list(APPEND SOURCES "src/Fisheye.cpp")

add_executable(fisheye main_fisheye.cpp ${SOURCES})
target_link_libraries(fisheye ${OpenCV_LIBS})
