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
#include <opencv2/core/eigen.hpp> //for using cv::cv2eigenssh nvidia@192.168.0.11 -Y
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
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_eigen/tf2_eigen.h>

#include <velodyne/pointcloudXYZIR.h>
#include <velodyne/point_types.h>
typedef velodyne_pcl::PointXYZIR PointT;

// caffe
#include "classifier_enet.h"
std::string model_path = "/home/nvidia/catkin_ws/src/kato_program/model/bn_conv_merged_model_kato1.prototxt";
std::string weights_path = "/home/nvidia/catkin_ws/src/kato_program/model/bn_conv_merged_weights_kato1.caffemodel";
std::string lut_path;
int flg_show_images;

#define LIDAR_LINE_NUM 32     //[本]
#define PI 3.141592

std::string IMAGE_TOPIC;
std::string POINTCLOUD_TOPIC;
std::string PUBLISH_FILTERED_POINTCLOUD_TOPIC;
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
double focal_x;
double focal_y;
double centre_x;
double centre_y;
double depth_filter_min;
double depth_filter_max;
double pass_filter_x;
double pass_filter_y_min;
double pass_filter_y_max;
// base_linkが存在するか確認
bool baseLinkExists = false;

cv::Mat global_enet_output;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
int state1=0;
int state2=0;
int filter_label_flg;

// caffe initialize
Classifier classifier(model_path, weights_path);

void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("camera_image_imput", IMAGE_TOPIC);
  nh.getParam("pointcloud_imput", POINTCLOUD_TOPIC);
  nh.getParam("extrinsic_matrix/data", extrinsic_matrix_1_vector);
  extrinsic_matrix_1 = (cv::Mat_<double>(4, 4) << extrinsic_matrix_1_vector[0], extrinsic_matrix_1_vector[1], extrinsic_matrix_1_vector[2], extrinsic_matrix_1_vector[3], extrinsic_matrix_1_vector[4], extrinsic_matrix_1_vector[5], extrinsic_matrix_1_vector[6], extrinsic_matrix_1_vector[7], extrinsic_matrix_1_vector[8], extrinsic_matrix_1_vector[9], extrinsic_matrix_1_vector[10], extrinsic_matrix_1_vector[11], extrinsic_matrix_1_vector[12], extrinsic_matrix_1_vector[13], extrinsic_matrix_1_vector[14], extrinsic_matrix_1_vector[15]);
  nh.getParam("camera_matrix/data", intrinsics_vector);
  camera_matrix_1 = (cv::Mat_<double>(3, 3) << intrinsics_vector[0], intrinsics_vector[1], intrinsics_vector[2], intrinsics_vector[3], intrinsics_vector[4], intrinsics_vector[5], intrinsics_vector[6], intrinsics_vector[7], intrinsics_vector[8]);
  focal_x = intrinsics_vector[0];
  focal_y = intrinsics_vector[4];
  centre_x = intrinsics_vector[2];
  centre_y = intrinsics_vector[5];
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
  nh.getParam("depth_filter_min", depth_filter_min);
  nh.getParam("depth_filter_max", depth_filter_max);
  nh.getParam("labeled_pointcloud_pass_filtered_name", PUBLISH_FILTERED_POINTCLOUD_TOPIC);
  nh.getParam("pointcloud_pass_through_x", pass_filter_x);
  nh.getParam("pointcloud_pass_through_y_min", pass_filter_y_min);
  nh.getParam("pointcloud_pass_through_y_max", pass_filter_y_max);
  nh.getParam("filter_label_flg", filter_label_flg);
}

Eigen::Matrix4d lidar2image_matrix(double offset_x, double offset_y, double offset_z, double offset_x_angle, double offset_y_angle, double offset_z_angle){
  cv::Mat vector = (cv::Mat_<double>(3, 1) << offset_x, offset_y, offset_z);
  cv::Mat rot_x = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
                0.0, cos(offset_x_angle), -sin(offset_x_angle),
                 0.0, sin(offset_x_angle), cos(offset_x_angle));
  cv::Mat rot_y = (cv::Mat_<double>(3, 3) << cos(offset_y_angle), 0.0, sin(offset_y_angle),
                 0.0, 1.0, 0.0,
                 -sin(offset_y_angle), 0.0, cos(offset_y_angle));
  cv::Mat rot_z = (cv::Mat_<double>(3, 3) << cos(offset_z_angle), -sin(offset_z_angle), 0.0,
                 sin(offset_z_angle), cos(offset_z_angle), 0.0,
                 0.0, 0.0, 1.0);
  cv::Mat rotation = rot_x.t() * rot_y.t() * rot_z.t();
  cv::Mat translation = vector;
  Eigen::Matrix4d eigenMatrix = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d rotationEigen;
  cv::cv2eigen(rotation, rotationEigen);
  eigenMatrix.block<3, 3>(0, 0) = rotationEigen;
  Eigen::Vector3d translationEigen;
  cv::cv2eigen(translation, translationEigen);
  eigenMatrix.block<3, 1>(0, 3) = translationEigen;
  return eigenMatrix;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv_bridge::CvImagePtr cvImage = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat distortedImage = cvImage->image;
    // cv::imshow("DistortedImage", distortedImage);
    // cv::waitKey(1);
    caffe::Blob<float> *output_layer;
    output_layer = classifier.Predict(distortedImage);
    cv::Mat enet_output = classifier.Visualization2(output_layer);
    cv::resize(enet_output, enet_output, cv::Size(), ORIGINAL_WIDTH_VALUE/enet_output.cols ,ORIGINAL_HEIGHT_VALUE/enet_output.rows, cv::INTER_NEAREST);
    cv::Mat undistorted_enet_output;
    //cv::remap(enet_output, undistorted_enet_output, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
    cv::remap(enet_output, undistorted_enet_output, map1, map2, cv::INTER_NEAREST);
    state1 +=1;
    global_enet_output = undistorted_enet_output;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_in)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_in, *cloud_ptr);
    global_pointcloud = cloud_ptr;
    state2 += 1;
}

// void filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
//                                                        int r_value, int g_value, int b_value) //x=4294967295白 4294901760赤 4294967040黄 4278190335青
// {
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

//     pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_condition(new pcl::ConditionAnd<pcl::PointXYZRGB>());
//     color_condition->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
//         new pcl::PackedRGBComparison<pcl::PointXYZRGB>("r", pcl::ComparisonOps::EQ, r_value)
//     ));
//     color_condition->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
//         new pcl::PackedRGBComparison<pcl::PointXYZRGB>("g", pcl::ComparisonOps::EQ, g_value)
//     ));
//     color_condition->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
//         new pcl::PackedRGBComparison<pcl::PointXYZRGB>("b", pcl::ComparisonOps::EQ, b_value)
//     ));
//     pcl::ConditionalRemoval<pcl::PointXYZRGB> color_filter;
//     color_filter.setInputCloud(cloud);
//     color_filter.setCondition(color_condition);
//     color_filter.setKeepOrganized(true);
//     color_filter.filter(*filtered_cloud);
//     *cloud = *filtered_cloud;
// }

// ==================================================
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "enet_lidar_fusion2_with_odometry_filtered");
    ros::NodeHandle nh;
    image_transport::ImageTransport transporter(nh);
    SetParameters();
    ros::Subscriber sbs_image = nh.subscribe(IMAGE_TOPIC, 1, imageCallback);
    ros::Subscriber sbs_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>(POINTCLOUD_TOPIC, 1, pointcloudCallback);
    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_FILTERED_POINTCLOUD_TOPIC, 1);
    Eigen::Matrix4d eigen_matrix = lidar2image_matrix(offset_vector[0], offset_vector[1], offset_vector[2], offset_vector[3] * PI / 180, offset_vector[4] * PI / 180, offset_vector[5] * PI / 180);
    Eigen::Matrix4d eigen_matrix_inverse = eigen_matrix.inverse();
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener(tf_buffer);
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        if (baseLinkExists == false){
          geometry_msgs::TransformStamped transformStamped2;
          try {
          transformStamped2 = tf_buffer.lookupTransform("map", "base_link", ros::Time(0));
          baseLinkExists = true;
          } catch(tf2::TransformException &ex) {
          // エラーが発生した場合、baseLinkExistsはfalseのままとなります
          ROS_WARN("%s", ex.what());
          }
        }
        if(state1 >= 1 && state2 >= 1 && baseLinkExists == true){
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_main(new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
          *cloud_main = *global_pointcloud;
          pcl::PassThrough<pcl::PointXYZRGB> pass;
          pass.setInputCloud(cloud_main);
          pass.setFilterFieldName("y");
          pass.setFilterLimits(-pass_filter_x, pass_filter_x);
          pass.filter(*cloud_main);
          pass.setInputCloud(cloud_main);
          pass.setFilterFieldName("x");
          pass.setFilterLimits(pass_filter_y_min, pass_filter_y_max);
          pass.filter(*cloud_main);
          pcl::transformPointCloud(*cloud_main, *cloud_main, eigen_matrix);
          for (size_t i=0; i<cloud_main->size();i++){
            float z = cloud_main->points[i].y;
            int pixel_pos_x;
            int pixel_pos_y;
            if(z!=0){
              float u = (-cloud_main->points[i].x*focal_x) / z;
              float v = (cloud_main->points[i].z*focal_y) / z;
              pixel_pos_x = (int)(u + centre_x);
              pixel_pos_y = (int)(v + centre_y);
            }
            cloud_main->points[i].r = 255;
            cloud_main->points[i].g = 255;
            cloud_main->points[i].b = 255;
            if(pixel_pos_x >= 0 && pixel_pos_x < global_enet_output.cols && pixel_pos_y >= 0 && pixel_pos_y < global_enet_output.rows && z > depth_filter_min && z < depth_filter_max){
              uchar value = global_enet_output.at<uchar>(pixel_pos_y, pixel_pos_x);
              if (value == 1){
                cloud_main->points[i].r = 255;
                cloud_main->points[i].g = 255;
                cloud_main->points[i].b = 0;
                if (filter_label_flg == 1 || filter_label_flg == 2){
                  filtered_cloud->points.push_back(cloud_main->points[i]);
                }
              }
              else if (value == 2){
                cloud_main->points[i].r = 0;
                cloud_main->points[i].g = 0;
                cloud_main->points[i].b = 255;
                if (filter_label_flg == 1 || filter_label_flg == 2){
                  filtered_cloud->points.push_back(cloud_main->points[i]);
                }
              }
              else if (value == 3){
                cloud_main->points[i].r = 255;
                cloud_main->points[i].g = 0;
                cloud_main->points[i].b = 0;
                if (filter_label_flg == 1){
                  filtered_cloud->points.push_back(cloud_main->points[i]);
                }
              }
            }
          }
          geometry_msgs::TransformStamped transformStamped;
          transformStamped = tf_buffer.lookupTransform("map", "base_link", ros::Time(0));

          Eigen::Quaterniond eigenQuat(transformStamped.transform.rotation.w, transformStamped.transform.rotation.x, transformStamped.transform.rotation.y, transformStamped.transform.rotation.z);

          // 平行移動ベクトルをEigen::Vector3dに変換
          Eigen::Vector3d eigenTranslation(transformStamped.transform.translation.x,
                                           transformStamped.transform.translation.y,
                                           transformStamped.transform.translation.z);

          // Eigen4d行列を作成
          Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Identity();
          transform_matrix.block<3, 3>(0, 0) = eigenQuat.toRotationMatrix();
          transform_matrix.block<3, 1>(0, 3) = eigenTranslation;
          transform_matrix = transform_matrix * eigen_matrix_inverse;
          if(filter_label_flg == 0)   {
            pcl::transformPointCloud(*cloud_main, *cloud_main, transform_matrix);
          // viewing
          sensor_msgs::PointCloud2 pub_pcl;
          pcl::toROSMsg(*cloud_main, pub_pcl);
          pub_pcl.header.frame_id = "map";
          pcl_pub.publish(pub_pcl);
          }
          else{
            pcl::transformPointCloud(*filtered_cloud, *filtered_cloud, transform_matrix);
          // viewing
          sensor_msgs::PointCloud2 pub_pcl;
          pcl::toROSMsg(*filtered_cloud, pub_pcl);
          pub_pcl.header.frame_id = "map";
          pcl_pub.publish(pub_pcl);
          }                         
        }
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
