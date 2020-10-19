cmake_minimum_required(VERSION 2.8.3)
project(stitch)

add_compile_options(-std=c++11)
add_compile_options(-O2)
add_compile_options(-pthread)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(OpenCV REQUIRED)

include_directories("include")

add_executable(main main.cpp src/Stitch.cpp src/imgops.cpp src/Contour.cpp)
target_link_libraries(main ${OpenCV_LIBS})
