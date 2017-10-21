cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(MY_GRAND_PROJECT)
find_package(PCL 1.3 REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


















#Add the source files and executables.
add_executable(pcd_write_test src/test.cpp)
target_link_libraries(pcd_write_test ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

add_executable(stereo_skin src/multitracker.cpp)
target_link_libraries(stereo_skin ${OpenCV_LIBS} ${PCL_LIBRARIES})
