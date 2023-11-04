cmake_minimum_required(VERSION 3.0.2)
project(kato_program)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  rospy
  image_transport
  pcl_conversions
  pcl_ros
  tf
  tf2
  tf2_ros
)

find_package(PkgConfig)
pkg_check_modules(OpenCV opencv4)
pkg_check_modules(CV_Bridge cv_bridge)

set(CAFFE_INCLUDEDIR /home/nvidia/e-net/ENet/caffe-enet/include /home/nvidia/e-net/ENet/caffe-enet/build/include)
set(CAFFE_LINK_LIBRARAY /home/nvidia/e-net/ENet/caffe-enet/build/lib)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(CUDA 10.0 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CV_Bridge_INCLUDE_DIRS}
  include/${PROJECT_NAME}
  ${CAFFE_INCLUDEDIR}
  ${CUDA_INCLUDE_DIRS}
  /usr/local/cuda/targets/${CUDA_TARGET_ARCH}/include
  include/${PROJECT_NAME}
  include
  )

link_directories(
  ${OpenCV_LIBRARY_DIRS}
  ${CV_Bridge_LIBRARY_DIRS}
  ${CAFFE_LINK_LIBRARAY}
  )

#include_directories(include ${catkin_INCLUDE_DIRS} )

#include_directories(
# include
# ${OpenCV_INCLUDE_DIRS}
#)

catkin_package()

#catkin_package(
#  INCLUDE_DIRS include
#  DEPENDS PCL GTSAM

#  CATKIN_DEPENDS
#  roscpp
#  pcl_conversions
#  pcl_ros
#)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_kato_program.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

add_executable(camera_enet_play
  src/camera_enet_play.cpp src/classifier_enet.cpp)
target_link_libraries(camera_enet_play
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CV_Bridge_LIBRARIES}
  caffe
  glog
  )

add_executable(camera_read
  src/camera_read.cpp)
target_link_libraries(camera_read
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CV_Bridge_LIBRARIES}
  )

add_executable(camera_read_compress
  src/camera_read_compress.cpp)
target_link_libraries(camera_read_compress
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CV_Bridge_LIBRARIES}
  )

add_executable(sub_compressed_image
  src/sub_compressed_image.cpp)
target_link_libraries(sub_compressed_image
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CV_Bridge_LIBRARIES}
  )

add_executable(camera_read_enet
  src/camera_read_enet.cpp src/classifier_enet.cpp)
target_link_libraries(camera_read_enet
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CV_Bridge_LIBRARIES}
  caffe
  glog
  )

add_executable(camera_lidar_fusion src/fusion/camera_lidar_fusion.cpp src/fusion/camera_info.hpp)
target_link_libraries(camera_lidar_fusion
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES})

add_executable(camera_lidar_fusion_manual src/fusion/camera_lidar_fusion_manual.cpp src/fusion/camera_info.hpp)
target_link_libraries(camera_lidar_fusion_manual
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES})

add_executable(enet_lidar_fusion src/fusion/enet_lidar_fusion.cpp src/fusion/camera_info.hpp src/classifier_enet.cpp)
target_link_libraries(enet_lidar_fusion
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES}
caffe
glog)

add_executable(enet_lidar_fusion_manual src/fusion/enet_lidar_fusion_manual.cpp src/fusion/camera_info.hpp src/classifier_enet.cpp)
target_link_libraries(enet_lidar_fusion_manual
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES}
caffe
glog)

add_executable(camera_lidar_fusion2 src/fusion/camera_lidar_fusion2.cpp)
target_link_libraries(camera_lidar_fusion2
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES})

add_executable(enet_lidar_fusion2 src/fusion/enet_lidar_fusion2.cpp src/classifier_enet.cpp)
target_link_libraries(enet_lidar_fusion2
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES}
caffe
glog)

add_executable(enet_lidar_fusion2_with_odometry src/fusion/enet_lidar_fusion2_with_odometry.cpp src/classifier_enet.cpp)
target_link_libraries(enet_lidar_fusion2_with_odometry
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES}
caffe
glog)

add_executable(enet_lidar_fusion2_with_odometry_filtered src/fusion/enet_lidar_fusion2_with_odometry_filtered.cpp src/classifier_enet.cpp)
target_link_libraries(enet_lidar_fusion2_with_odometry_filtered
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES}
caffe
glog)

add_executable(global_labeled_pointcloud src/fusion/global_labeled_pointcloud.cpp)
target_link_libraries(global_labeled_pointcloud
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}
${CV_Bridge_LIBRARIES})

add_executable(matching src/control/matching.cpp)
target_link_libraries(matching
${catkin_LIBRARIES} 
${CV_Bridge_LIBRARIES})

add_executable(control src/control/control.cpp)
target_link_libraries(control
${catkin_LIBRARIES}
${CV_Bridge_LIBRARIES})