project(general)
cmake_minimum_required(VERSION 2.8.3)

macro(find_opencv)
  if(WIN32)
    # put opencv_world???.dll near the final executable before running
    # which is located at "${OpenCV_DIR}/../bin"
    # or add "${OpenCV_DIR}/../bin" to PATH environment variable
    # cmake -D OpenCV_DIR="C:/opencv/build/x64/vc15/lib" ..
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif(WIN32)

  find_package(OpenCV REQUIRED)

  if (WIN32 AND NOT BUILD_SHARED_LIBS)
    # MSVC settings for static build
    # https://stackoverflow.com/a/14172871/12447766
    set(CompilerFlags
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
    )
    foreach(CompilerFlag ${CompilerFlags})
      string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
  endif(WIN32 AND NOT BUILD_SHARED_LIBS)

  set(CMAKE_CXX_STANDARD 14)
  if(NOT WIN32)
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-pthread)
  endif(NOT WIN32)
endmacro(find_opencv)

find_opencv()

set(SOURCES "src/imgops.cpp" "src/Contour.cpp")

if(WIN32)
  add_library(general_library STATIC ${SOURCES})
else(WIN32)
  add_library(general_library SHARED ${SOURCES})
endif(WIN32)

target_link_libraries(general_library ${OpenCV_LIBS})
target_include_directories(general_library PUBLIC "include")

option(BUILD_SUBDIRECTORIES "Build all projects" OFF)
message(STATUS "Build subdirectories: " ${BUILD_SUBDIRECTORIES})
if(BUILD_SUBDIRECTORIES)
  if(NOT TARGET stitch_library)
    add_subdirectory(Stitch)
  endif(NOT TARGET stitch_library)
  add_subdirectory(Calibrate)
  add_subdirectory(Jpeg)
  add_subdirectory(Segment)
endif(BUILD_SUBDIRECTORIES)

include_directories("include")

add_executable(optflow src/optflow.cpp)
target_link_libraries(optflow general_library)

add_executable(tracker src/tracker.cpp)
target_link_libraries(tracker general_library)
