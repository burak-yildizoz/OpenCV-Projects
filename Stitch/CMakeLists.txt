cmake_minimum_required(VERSION 2.8.3)
project(stitch)

if(WIN32)
  # put opencv_world???.dll near the final executable before running
  # which is located at "${OpenCV_DIR}/../bin"
  set(OpenCV_DIR "C:/opencv/build/x64/vc15/lib")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(WIN32)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
if(NOT WIN32)
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-pthread)
endif(NOT WIN32)

set(SOURCES "src/Stitch.cpp" "src/imgops.cpp" "src/Contour.cpp")

add_library(stitch SHARED ${SOURCES})
target_link_libraries(stitch ${OpenCV_LIBS})
target_include_directories(stitch PUBLIC ${CMAKE_CURRENT_LIST_DIR})

include_directories("include")

add_executable(main main.cpp ${SOURCES})
target_link_libraries(main stitch)

add_executable(patch_demo patch_demo.cpp ${SOURCES})
target_link_libraries(patch_demo stitch)

add_executable(combine_parts combine_parts.cpp ${SOURCES})
target_link_libraries(combine_parts stitch)
