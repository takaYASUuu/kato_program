/**
   @date  Time-stamp: <2020-09-18 14:47:28 nvidia>
   @brief test program for topic communication.
**/

#include <iostream>
#include <sstream> // stringstream
#include <deque>   //キューを使用する時
using namespace std;

#include <Eigen/Core>

#include <opencv2/opencv.hpp>     // Include OpenCV API
#include <opencv2/core/eigen.hpp> //for using cv::cv2eigen
// using namespace cv;

#include <omp.h>
#include <mutex>
#include <chrono>
#include <algorithm> //std::sort

// ROS
#include <ros/ros.h>
#include <std_msgs/String.h> // image_name
#include <std_msgs/Float64MultiArray.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Twist.h>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <velodyne/pointcloudXYZIR.h>
#include <velodyne/point_types.h>

// typedef pcl::PointXYZI PointT;
typedef velodyne_pcl::PointXYZIR PointT;

#include "camera_info.hpp"

pcl::PointCloud<PointT>::Ptr global_pointcloud_xyzir_ptr(new pcl::PointCloud<PointT>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_pointcloud_xyzrgb_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

#define LIDAR_LINE_NUM 32     //[本]
#define PI 3.141592

std::string IMAGE_TOPIC;
std::string POINTCLOUD_TOPIC;
cv::Mat extrinsic_matrix_1;
cv::Mat camera_matrix_1;
cv::Mat new_camera_matrix_1;
cv::Mat distortion_1;
std::vector<double> extrinsic_matrix_1_vector;
std::vector<double> intrinsics_vector;
std::vector<double> new_camera_matrix_1_vector;
std::vector<double> distortionCoeffs_vector;
std::vector<double> offset_vector;
cv::Size2i image_size_1;
double ORIGINAL_WIDTH_VALUE;
double ORIGINAL_HEIGHT_VALUE;
cv::Mat map1;
cv::Mat map2;
camera_param camera1;

cv::Mat global_undistortedImage;
pcl::PointCloud<PointT>::Ptr global_pointcloud(new pcl::PointCloud<PointT>);
int state1=0;
int state2=0;

void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("camera_image_imput", IMAGE_TOPIC);
  nh.getParam("pointcloud_imput", POINTCLOUD_TOPIC);
  nh.getParam("extrinsic_matrix/data", extrinsic_matrix_1_vector);
  extrinsic_matrix_1 = (cv::Mat_<double>(4, 4) << extrinsic_matrix_1_vector[0], extrinsic_matrix_1_vector[1], extrinsic_matrix_1_vector[2], extrinsic_matrix_1_vector[3], extrinsic_matrix_1_vector[4], extrinsic_matrix_1_vector[5], extrinsic_matrix_1_vector[6], extrinsic_matrix_1_vector[7], extrinsic_matrix_1_vector[8], extrinsic_matrix_1_vector[9], extrinsic_matrix_1_vector[10], extrinsic_matrix_1_vector[11], extrinsic_matrix_1_vector[12], extrinsic_matrix_1_vector[13], extrinsic_matrix_1_vector[14], extrinsic_matrix_1_vector[15]);
  nh.getParam("camera_matrix/data", intrinsics_vector);
  camera_matrix_1 = (cv::Mat_<double>(3, 3) << intrinsics_vector[0], intrinsics_vector[1], intrinsics_vector[2], intrinsics_vector[3], intrinsics_vector[4], intrinsics_vector[5], intrinsics_vector[6], intrinsics_vector[7], intrinsics_vector[8]);
  nh.getParam("projection_matrix/data", new_camera_matrix_1_vector);
  new_camera_matrix_1 = (cv::Mat_<double>(3, 3) << new_camera_matrix_1_vector[0], new_camera_matrix_1_vector[1], new_camera_matrix_1_vector[2], new_camera_matrix_1_vector[3], new_camera_matrix_1_vector[4], new_camera_matrix_1_vector[5], new_camera_matrix_1_vector[6], new_camera_matrix_1_vector[7], new_camera_matrix_1_vector[8]);
  nh.getParam("distortion_coefficients/data", distortionCoeffs_vector);
  distortion_1 = (cv::Mat_<double>(1, distortionCoeffs_vector.size()) << distortionCoeffs_vector[0], distortionCoeffs_vector[1], distortionCoeffs_vector[2], distortionCoeffs_vector[3], distortionCoeffs_vector[4]);
  nh.getParam("original_width", ORIGINAL_WIDTH_VALUE);
  nh.getParam("original_height", ORIGINAL_HEIGHT_VALUE);
  image_size_1.width = (int) ORIGINAL_WIDTH_VALUE;
  image_size_1.height = (int) ORIGINAL_HEIGHT_VALUE;
  new_camera_matrix_1 = cv::getOptimalNewCameraMatrix(camera_matrix_1, distortion_1, image_size_1, 1, image_size_1);
  cv::initUndistortRectifyMap(camera_matrix_1, distortion_1, cv::Mat(), new_camera_matrix_1, image_size_1, CV_32FC1, map1, map2);
  nh.getParam("offset_parameter", offset_vector);
}

Eigen::MatrixXd lidar_to_image2_matrix(vector<cv::Point3d> point_array, camera_param camera)
{
  // undistort image, newCameraMatrixに点を乗せる
  cv::Mat point_matrix_cv = cv::Mat(point_array).reshape(1).t();
  Eigen::MatrixXd point_matrix_eigen;
  cv::cv2eigen(point_matrix_cv, point_matrix_eigen);

  Eigen::MatrixXd ret;
  ret.resize(2, point_matrix_eigen.cols());
  Eigen::Matrix<double, 3, 3> new_camera_mat;
  cv::cv2eigen(camera.newCameraMatrix, new_camera_mat);
  Eigen::MatrixXd s;
  s.resize(1, point_matrix_eigen.cols());
  Eigen::MatrixXd mat1;
  Eigen::MatrixXd mat2;
  Eigen::MatrixXd mat2_array;
  mat2_array.resize(3, point_matrix_eigen.cols());
  Eigen::MatrixXd mat_tmp;
  mat1 = new_camera_mat * camera.rotation_mat * point_matrix_eigen;
  mat2 = new_camera_mat * camera.translation_vec;
  for (int i = 0; i < point_matrix_eigen.cols(); i++)
  {
    mat2_array.col(i) = mat2.col(0);
  }
  mat_tmp = mat1 + mat2_array;
  s.row(0) = mat_tmp.row(2);
  ret.row(0) = mat_tmp.row(0).array() / s.row(0).array();
  ret.row(1) = mat_tmp.row(1).array() / s.row(0).array();

  return ret;
}

cv::Mat lidar_on_image(cv::Mat image, pcl::PointCloud<PointT> cloud, camera_param camera)
{
  int point_num = cloud.points.size();
  if (point_num != 0)
  {
    vector<cv::Point3d> point_cloud_array;
    vector<float> point_intensity_array;
    vector<uint16_t> point_ring_array;
    for (int j = 0; j < point_num; j++)
    {
      cv::Point3d point_cv = {cloud.points[j].x, cloud.points[j].y, cloud.points[j].z};
      float point_intensity = cloud.points[j].intensity;
      uint16_t point_ring = cloud.points[j].ring;

      if (point_cv.x >= 0)
      {
        point_cloud_array.push_back(point_cv);
        point_intensity_array.push_back(point_intensity);
        point_ring_array.push_back(point_ring);
      }
    }
    Eigen::MatrixXd image_point_mat = lidar_to_image2_matrix(point_cloud_array, camera);

    for (int i = 0; i < image_point_mat.cols(); i++)
    {
      cv::Point2d pcl_point;
      pcl_point.x = image_point_mat(0, i);
      pcl_point.y = image_point_mat(1, i);
      float point_intensity = point_intensity_array.at(i);
      uint16_t point_ring = point_ring_array.at(i);
      cv::circle(image, pcl_point, 2, cv::Scalar(point_ring * 255 / LIDAR_LINE_NUM, 2 * point_intensity, 0), -1);
    }
  }
  return image;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv_bridge::CvImagePtr cvImage = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat distortedImage = cvImage->image;
    // cv::imshow("DistortedImage", distortedImage);
    // cv::waitKey(1);

     // 歪み補正
    cv::Mat undistortedImage;
    state1 +=1;
    cv::remap(distortedImage, undistortedImage, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
    global_undistortedImage = undistortedImage;

    // 歪み補正された画像を表示
    // cv::imshow("UndistortedImage", undistortedImage);
    // cv::waitKey(1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_in)
{
    pcl::PointCloud<PointT>::Ptr cloud_ptr(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_in, *cloud_ptr);
    global_pointcloud = cloud_ptr;
    state2 += 1;
}

// ==================================================
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "camera_lidar_fusion");
    ros::NodeHandle nh;
    image_transport::ImageTransport transporter(nh);
    SetParameters();

    // subscribe topic
    // ros::Subscriber sbs_gmsl_image = nh.subscribe("/gmsl/image_0", 1, image_callback);
    // ros::Subscriber sbs_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, pointcloud_callback);

    //カメラ行列の読み込み
    // camera1.set(extrinsic_matrix_1, camera_matrix_1, distortion_1, image_size_1);
    camera1.set(extrinsic_matrix_1, camera_matrix_1, new_camera_matrix_1, distortion_1, image_size_1);
    ros::Subscriber sbs_image = nh.subscribe(IMAGE_TOPIC, 1, imageCallback);
    ros::Subscriber sbs_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>(POINTCLOUD_TOPIC, 1, pointcloudCallback);
    camera1.offset(offset_vector[0], offset_vector[1], offset_vector[2], offset_vector[3] * PI / 180, offset_vector[4] * PI / 180, offset_vector[5] * PI / 180);
    //camera1.offset(0, 0.5, 0.3, -3.5 * PI / 180);
    // camera1.print();

    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        if(state1 >= 1 && state2 >= 1){
            // std::cout<< "bbb" << std::endl;
            // std::cout<< global_undistortedImage << std::endl;
        cv::imshow("camera1", lidar_on_image(global_undistortedImage, *global_pointcloud, camera1));
        cv::waitKey(1);
        }
        //lidar_on_image(global_undistortedImage, *global_pointcloud, camera1);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
