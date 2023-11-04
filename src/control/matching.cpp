#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include <Eigen/Core>


std::string TRAJECTORY_TOPIC;
bool baseLinkExists = false;
bool init_trajectory_flg = false;
bool set_5s_flag = false;
bool set_5s_end_flg = false;
bool tmp_pos = false;
pcl::PointCloud<pcl::PointXYZ>::Ptr trajectory_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_preliminary_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
int global_position_index;
std::vector<int> global_position_vector;
std::string POSITION_TOPIC;
std::string GLOBAL_POINTCLOUD_TOPIC;
int max_vector_size;
double filter_range;
int look_far_range;


void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("trajectory_topic", TRAJECTORY_TOPIC);
  nh.getParam("position_index_topic", POSITION_TOPIC);
  nh.getParam("global_labeled_pointcloud", GLOBAL_POINTCLOUD_TOPIC);
  nh.getParam("max_vector_size", max_vector_size);
  nh.getParam("filter_range", filter_range);
  nh.getParam("look_far_range", look_far_range);
}

void position_callback(const std_msgs::Int32::ConstPtr& msg) {
  global_position_index = msg->data;
  if(global_position_index <= 100 && tmp_pos == false){
    global_position_vector = {100, 120, 140, 160, 180, 200};
    tmp_pos = true;
  }
  else if(global_position_index > 220 - look_far_range){
    if (std::find(global_position_vector.begin(), global_position_vector.end(), (global_position_index/20) * 20 + look_far_range) == global_position_vector.end()) {
        global_position_vector.push_back((global_position_index/20) * 20 + look_far_range);
    }
  }
  if (global_position_vector.size() > max_vector_size) {
    global_position_vector.erase(global_position_vector.begin());
    }
}

void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_in)
{
    pcl::fromROSMsg(*cloud_in, *global_cloud);
}

std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pass_through(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud, double x, double y, double range){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(x-range, x+range);
  pass.filter(*filtered_cloud);
  pass.setNegative(true);
  pass.filter(*remaining_cloud);
  pass.setNegative(false);

  // y軸方向の範囲フィルタリング
  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(y-range, y+range);
  pass.filter(*filtered_cloud);
  pass.setNegative(true);
  pass.filter(*remaining_cloud);
  return std::make_pair(filtered_cloud, remaining_cloud);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pass_through2(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud, double range){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr return_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  bool tmp_pass;
  std::vector<int> position_vector = global_position_vector;
  for (size_t i=0; i<input_cloud->size();i++){
    tmp_pass = false;
    for (int num : position_vector) {
      if(tmp_pass == false && (trajectory_cloud->points[num].x - input_cloud->points[i].x)*(trajectory_cloud->points[num].x - input_cloud->points[i].x) + (trajectory_cloud->points[num].y - input_cloud->points[i].y)*(trajectory_cloud->points[num].y - input_cloud->points[i].y) < range * range){
        tmp_pass = true;
        return_cloud->points.push_back(input_cloud->points[i]);
      }
    }
  }
  return return_cloud;
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "matching");
  ros::NodeHandle nh;
  SetParameters();
  pcl::io::loadPCDFile<pcl::PointXYZ>("/home/nvidia/catkin_ws/src/kato_program/map/preliminary_trajectory/smoothed_trajectory.pcd", *trajectory_cloud);
  pcl::io::loadPCDFile<pcl::PointXYZRGB>("/home/nvidia/catkin_ws/src/kato_program/map/preliminary_map/preliminary_map.pcd", *global_preliminary_cloud);
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);
  ros::Time::waitForValid();
  ros::WallTime wall_begin;
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(TRAJECTORY_TOPIC, 1);
  ros::Publisher pub1 = nh.advertise<sensor_msgs::PointCloud2>("tmp_cloud1", 1);
  ros::Publisher pub2 = nh.advertise<sensor_msgs::PointCloud2>("tmp_cloud2", 1);
  ros::Subscriber sub_position = nh.subscribe(POSITION_TOPIC, 1, position_callback);
  ros::Subscriber sub_pointcloud = nh.subscribe(GLOBAL_POINTCLOUD_TOPIC, 1, pointcloudCallback);
  ros::Rate loop_rate(30);
  while (ros::ok())
  {
    if (baseLinkExists == false){
          geometry_msgs::TransformStamped transformStamped2;
          try {
          transformStamped2 = tf_buffer.lookupTransform("map", "base_link", ros::Time(0));
          baseLinkExists = true;
          wall_begin = ros::WallTime::now();
          } catch(tf2::TransformException &ex) {
          // エラーが発生した場合、baseLinkExistsはfalseのままとなります
          ROS_WARN("%s", ex.what());
          }
    }
    if (baseLinkExists){
      if (init_trajectory_flg == false){
        tf2::Vector3 start_point(trajectory_cloud->points[0].x, trajectory_cloud->points[0].y, trajectory_cloud->points[0].z); // 始点の座標 [x, y, z]
        tf2::Vector3 end_point(trajectory_cloud->points[100].x, trajectory_cloud->points[100].y, trajectory_cloud->points[100].z); // 終点の座標 [x, y, z]
        tf2::Vector3 vector_init = end_point - start_point;
        geometry_msgs::TransformStamped transformStamped;
        transformStamped = tf_buffer.lookupTransform("map", "base_link", ros::Time(0));
        tf2::Matrix3x3 rotation_matrix_init(tf2::Quaternion(
        transformStamped.transform.rotation.x,
        transformStamped.transform.rotation.y,
        transformStamped.transform.rotation.z,
        transformStamped.transform.rotation.w));
        tf2::Vector3 x_axis(1, 0, 0);
        tf2::Vector3 base_link_x_axis = rotation_matrix_init * x_axis;
        double dot_product_init = vector_init.dot(base_link_x_axis);
        double norm_vector_init = vector_init.length();
        double norm_base_link_x_axis_init = base_link_x_axis.length();
        double angleRadians = std::acos(dot_product_init / (norm_vector_init * norm_base_link_x_axis_init));
        
        Eigen::Affine3f transformMatrix_init = Eigen::Affine3f::Identity();
        transformMatrix_init.rotate(Eigen::AngleAxisf(-angleRadians, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*trajectory_cloud, *trajectory_cloud, transformMatrix_init);
        pcl::transformPointCloud(*global_preliminary_cloud, *global_preliminary_cloud, transformMatrix_init);

        sensor_msgs::PointCloud2 output_cloud_msg;
        pcl::toROSMsg(*trajectory_cloud, output_cloud_msg);
        output_cloud_msg.header.frame_id = "map";
        pub.publish(output_cloud_msg);
        init_trajectory_flg = true;
      }
      else{
        if ((ros::WallTime::now() - wall_begin).sec > 4.0 && set_5s_end_flg == false){
          geometry_msgs::TransformStamped transformStamped;
          transformStamped = tf_buffer.lookupTransform("map", "base_link", ros::Time(0));
          tf2::Vector3 start_point_5s(0, 0, 0); // 始点の座標 [x, y, z]
          tf2::Vector3 end_point_5s(transformStamped.transform.translation.x,
                           transformStamped.transform.translation.y,
                           transformStamped.transform.translation.z); // 終点の座標 [x, y, z]
          tf2::Vector3 vector_5s = end_point_5s - start_point_5s;
          tf2::Vector3 start_point_5s_cloud(trajectory_cloud->points[0].x, trajectory_cloud->points[0].y, trajectory_cloud->points[0].z); // 始点の座標 [x, y, z]
          tf2::Vector3 end_point_5s_cloud(trajectory_cloud->points[100].x, trajectory_cloud->points[100].y, trajectory_cloud->points[100].z); // 終点の座標 [x, y, z]
          tf2::Vector3 vector_5s_cloud = end_point_5s_cloud - start_point_5s_cloud;
          double dot_product_5s = vector_5s_cloud.dot(vector_5s);
          double norm_vector_5s = vector_5s_cloud.length();
          double norm_base_link_x_5s = vector_5s.length();
          double angleRadians_5s = std::acos(dot_product_5s / (norm_vector_5s * norm_base_link_x_5s));

          Eigen::Affine3f transformMatrix_5s = Eigen::Affine3f::Identity();
          transformMatrix_5s.rotate(Eigen::AngleAxisf(angleRadians_5s, Eigen::Vector3f::UnitZ()));
          pcl::transformPointCloud(*trajectory_cloud, *trajectory_cloud, transformMatrix_5s);
          pcl::transformPointCloud(*global_preliminary_cloud, *global_preliminary_cloud, transformMatrix_5s);
          sensor_msgs::PointCloud2 output_cloud_msg;
          pcl::toROSMsg(*trajectory_cloud, output_cloud_msg);
          output_cloud_msg.header.frame_id = "map";
          pub.publish(output_cloud_msg);
          set_5s_end_flg = true;
        }
        else if(set_5s_end_flg && (ros::WallTime::now() - wall_begin).sec > 12.0){
          //pcl::PointCloud<pcl::PointXYZRGB>::Ptr preliminary_cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr preliminary_cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
          // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp(new pcl::Poin          tCloud<pcl::PointXYZRGB>);
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
          preliminary_cloud_filtered = pass_through2(global_preliminary_cloud, filter_range);
          cloud_filtered = pass_through2(global_cloud, filter_range);
          //preliminary_cloud_tmp = global_preliminary_cloud;
          // for (int num : global_position_vector) {
          //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr a_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
          //   std::tie(a_cloud, preliminary_cloud_tmp) = pass_through(preliminary_cloud_tmp, trajectory_cloud->points[num].x, trajectory_cloud->points[num].y, 2.0);
          //   *preliminary_cloud_filtered += *a_cloud;
          // }
          // cloud_tmp = global_cloud;
          // for (int num : global_position_vector) {
          //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr a_cloud(new pcl::PointCloud<pcl::Poin   tXYZRGB>);
          //   std::tie(a_cloud, cloud_tmp) = pass_through(cloud_tmp, trajectory_cloud->points[num].x, trajectory_cloud->points[num].y, 2.0);
          //   *cloud_filtered += *a_cloud;
          // }
          pcl::VoxelGrid<pcl::PointXYZRGB> sor;
          sor.setInputCloud(preliminary_cloud_filtered);
          sor.setLeafSize(1.0f, 1.0f, 1.0f); // ボクセルのサイズを設定
          sor.filter(*preliminary_cloud_filtered); // ボクセル化されたソース点群
          sor.setInputCloud(cloud_filtered);
          sor.filter(*cloud_filtered); // ボクセル化されたターゲット点群
          sensor_msgs::PointCloud2 output_cloud_msg1;
          pcl::toROSMsg(*preliminary_cloud_filtered, output_cloud_msg1);
          output_cloud_msg1.header.frame_id = "map";
          pub1.publish(output_cloud_msg1);
          sensor_msgs::PointCloud2 output_cloud_msg2;
          pcl::toROSMsg(*cloud_filtered, output_cloud_msg2);
          output_cloud_msg2.header.frame_id = "map";
          pub2.publish(output_cloud_msg2);
          pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
          icp.setInputSource(preliminary_cloud_filtered);
          icp.setInputTarget(cloud_filtered);
          icp.setMaximumIterations (70);
          //icp.setMaxCorrespondenceDistance(0.05)
          icp.setEuclideanFitnessEpsilon (0.0001);
          //icp.setUseReciprocalCorrespondences(true)
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
          icp.align(*aligned_cloud);
          Eigen::Matrix4f transformation = icp.getFinalTransformation();
          pcl::transformPointCloud(*trajectory_cloud, *trajectory_cloud, transformation);
          pcl::transformPointCloud(*global_preliminary_cloud, *global_preliminary_cloud, transformation);
          sensor_msgs::PointCloud2 output_cloud_msg;
          pcl::toROSMsg(*trajectory_cloud, output_cloud_msg);
          output_cloud_msg.header.frame_id = "map";
          pub.publish(output_cloud_msg);
        }
      }
    }
    
    ros::spinOnce();
    loop_rate.sleep();
  }
  // Create a ROS publisher to PUBLISH_TOPIC with a queue_size of 1
  //pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_TOPIC, 1);

  // Spin
  //ros::spin();

  // Success
  return 0;
}