cmake_minimum_required(VERSION 2.4.0)

project(egbis)
# https://github.com/christofferholmstedt/opencv-wrapper-egbis

find_package(OpenCV REQUIRED)

# Find source files
file(GLOB SOURCES egbis.cpp egbis/disjoint-set.cpp egbis/misc.cpp egbis/segment-graph.cpp egbis/filter.cpp egbis/segment-image.cpp)

add_library(egbis SHARED ${SOURCES})
target_link_libraries(egbis ${OpenCV_LIBS})
target_include_directories(egbis PUBLIC ${CMAKE_CURRENT_LIST_DIR})
