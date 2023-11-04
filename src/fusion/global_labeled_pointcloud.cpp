/**
   @date  Time-stamp: <2020-09-18 14:47:28 nvidia>
   @brief test program for topic communication.
**/

#include <iostream>
#include <sstream> // stringstream

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Twist.h>

#include <tf2_ros/transform_listener.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <velodyne/pointcloudXYZIR.h>
#include <velodyne/point_types.h>
#include <pcl/io/pcd_io.h>
typedef velodyne_pcl::PointXYZIR PointT;

#define LIDAR_LINE_NUM 32     //[本]
#define PI 3.141592

std::string LABELED_POINTCLOUD_TOPIC;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
int state1=0;
int state2=0;
int state_tmp = 1;
double global_base_link_x = 0;
double global_base_link_y = 0;
double global_tmp_base_link_x = 0;
double global_tmp_base_link_y = 0;
double save_distance_square;
bool savePointCloud = false;
std::string save_pcd_directory;

ros::Publisher pub;
std::string global_labeled_pointcloud_string;


void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("labeled_pointcloud_pass_filtered_name", LABELED_POINTCLOUD_TOPIC);
  nh.getParam("save_distance_square", save_distance_square);
  nh.getParam("save_pcd_directory", save_pcd_directory);
  nh.getParam("global_labeled_pointcloud", global_labeled_pointcloud_string);
}

void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_in)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_in, *cloud_ptr);
    if(state_tmp == 1){
      *global_pointcloud += *cloud_ptr;
      state1 = 1;
      state_tmp = 0;
    }
    sensor_msgs::PointCloud2 output_cloud_msg;
    pcl::toROSMsg(*global_pointcloud, output_cloud_msg);
    output_cloud_msg.header.frame_id = "map";

    // 新しいPoitCloud2メッセージをPublish
    pub.publish(output_cloud_msg);
}

// ==================================================
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "global_labeled_pointcloud");
    ros::NodeHandle nh;
    SetParameters();
    ros::Subscriber sbs_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>(LABELED_POINTCLOUD_TOPIC, 1, pointcloudCallback);
    //ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_POINTCLOUD_TOPIC, 1);
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener(tf_buffer);
    pub = nh.advertise<sensor_msgs::PointCloud2>(global_labeled_pointcloud_string, 1);
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
       try
        {
            // mapからbase_linkへの変換を取得
            geometry_msgs::TransformStamped transformStamped;
            transformStamped = tf_buffer.lookupTransform("map", "base_link", ros::Time(0));

            // base_linkのx座標とy座標を取得
            global_base_link_x = transformStamped.transform.translation.x;
            global_base_link_y = transformStamped.transform.translation.y;
            if(state1 == 1){
              if((global_base_link_x - global_tmp_base_link_x)*(global_base_link_x - global_tmp_base_link_x)+(global_base_link_y-global_tmp_base_link_y)*(global_base_link_y-global_tmp_base_link_y)>=save_distance_square){
                state_tmp = 1;
                global_tmp_base_link_x = global_base_link_x;
                global_tmp_base_link_y = global_base_link_y;
              }
            }        

        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s",ex.what());
            ros::Duration(1.0).sleep();
            continue;
        }
      ros::spinOnce();
      loop_rate.sleep();
    }
    pcl::io::savePCDFileBinary(save_pcd_directory, *global_pointcloud);

    return 0;
}
