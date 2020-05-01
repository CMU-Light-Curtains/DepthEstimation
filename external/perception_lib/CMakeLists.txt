cmake_minimum_required(VERSION 2.8.3)
project(perception_lib)


#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/perception/cmake/modules")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

#find_package(catkin REQUIRED COMPONENTS
#  pcl_conversions
#  pcl_ros
#  roscpp
#  sensor_msgs
#  dynamic_reconfigure
#  cv_bridge
#  image_geometry
#)

#catkin_package(
#    LIBRARIES perception_lib
#    INCLUDE_DIRS include
#)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB perception_lib_SRC
    "include/*.h"
    "include/*.hpp"
    "include/*.cl"
    "src/*.cpp"
    #"../vision_launch/launch/*"
    #"../vision_launch/scripts/*"
)

add_library(perception_lib SHARED ${perception_lib_SRC})
target_link_libraries(
  perception_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
set_target_properties(perception_lib PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(pybind11)

pybind11_add_module(pyperception_lib python/pyperception_lib.cpp)
target_link_libraries(pyperception_lib PRIVATE pybind11::module
    perception_lib ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${catkin_LIBRARIES}
)