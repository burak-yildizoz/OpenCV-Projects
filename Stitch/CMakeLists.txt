cmake_minimum_required(VERSION 2.8.3)
project(stitch)

if(NOT TARGET general_library)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/general EXCLUDE_FROM_ALL)
endif(NOT TARGET general_library)
find_opencv()

set(SOURCES "src/Stitch.cpp")

if(WIN32)
  add_library(stitch_library STATIC ${SOURCES})
else(WIN32)
  add_library(stitch_library SHARED ${SOURCES})
endif(WIN32)
target_link_libraries(stitch_library PUBLIC general_library)
target_include_directories(stitch_library PUBLIC "include")

include_directories("include")

add_executable(stitch main.cpp ${SOURCES})
target_link_libraries(stitch stitch_library)

add_executable(patch_demo patch_demo.cpp ${SOURCES})
target_link_libraries(patch_demo stitch_library)

add_executable(combine_parts combine_parts.cpp ${SOURCES})
target_link_libraries(combine_parts stitch_library)
