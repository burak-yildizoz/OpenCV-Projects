cmake_minimum_required(VERSION 2.8.3)
project(stitch)

add_compile_options(-std=c++11)
add_compile_options(-O2)
add_compile_options(-pthread)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(OpenCV REQUIRED)

include_directories("include")

set(SOURCES "src/Stitch.cpp" "src/imgops.cpp" "src/Contour.cpp")

add_executable(main main.cpp ${SOURCES})
target_link_libraries(main ${OpenCV_LIBS})

add_executable(patch_demo patch_demo.cpp ${SOURCES})
target_link_libraries(patch_demo ${OpenCV_LIBS})

add_executable(combine_parts combine_parts.cpp ${SOURCES})
target_link_libraries(combine_parts ${OpenCV_LIBS})

