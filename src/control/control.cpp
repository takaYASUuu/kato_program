#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include "std_msgs/Float64MultiArray.h"


//tf関連
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <turtlesim/Spawn.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

std::string TRAJECTORY_TOPIC;
std::string POSITION_TOPIC;
bool baseLinkExists = false;
bool init_trajectory_flg = false;
bool set_5s_flag = false;
bool set_5s_end_flg = false;
int state1 = 0;
std::vector< std::vector<double> > preliminary_path;
std::vector< std::vector<double> > point_vk;
ros::Publisher pub_position;
ros::Publisher pub_e;

double vr0;
double K2;
double K3;

double PI = 3.141592;
//最大曲率
double c_max = std::sin(PI*40/180) / 1.905; //sin40度が限界 1905はホイールベース(前輪と後輪の距離)

void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("trajectory_topic", TRAJECTORY_TOPIC);
  // nh.getParam("vr0_control", vr0);
  // nh.getParam("K2_control", K2);
  // nh.getParam("K3_control", K3);
  int a;
  //int b;
  std::vector<double> c;
  nh.getParam("point_vk/rows", a);
  //nh.getParam("point_vk/cols", b);
  nh.getParam("point_vk/data", c);
  for (int j = 0; j < a; j++){
    point_vk.push_back({c[j*4],c[j*4+1],c[j*4+2],c[j*4+3]});
  }
  nh.getParam("position_index_topic", POSITION_TOPIC);
}

double calcurvature(double ax, double ay, double bx, double by, double cx, double cy){
    double ab= pow(bx-ax,2) + pow(by-ay,2);
    double bc= pow(cx-bx,2) + pow(cy-by,2);
    double innerProduct = (bx - ax) * (bx - cx) + (by - ay) * (by - cy);
    double cos2_theta = pow(innerProduct, 2) / (ab * bc);
    if (abs(cos2_theta) > 0.99999999){
        return 0;
    }
     double sin_theta = sqrt(1 - pow(innerProduct, 2) / (ab * bc));

     double outerProduct = (bx - ax) * (by - cy) - (by - ay) * (bx - cx);
     if (outerProduct > 0){
        sin_theta *= -1;
    }
    return 2 * sin_theta / sqrt(pow(cx-ax,2) + pow(cy-ay,2));
}

int closest_point_index (double x, double y, std::vector< std::vector<double> > array){
    double tmp = pow(x-array[0][0],2) + pow(y-array[0][1],2);
    int tmp_number = 0;
    for (int i=0; i < array.size(); i++){
        if (tmp > pow(x-array[i][0],2) + pow(y-array[i][1],2)){
            tmp = pow(x-array[i][0],2) + pow(y-array[i][1],2);
            tmp_number = i;
        }
    }
    return tmp_number;
}

std::vector< std::vector<double> > add_curvature(std::vector< std::vector<double> > array){
    std::vector< std::vector<double> > array2;
    for (int i=0; i < array.size(); i++){
        if(i==0){
            array2.push_back({array[i][0],array[i][1],0});
        }
        else if (i==array.size()-1 || i==array.size()-2){
            array2.push_back({array[i][0],array[i][1],0});
        }
        else{
			if(calcurvature(array[i-1][0],array[i-1][1],array[i][0],array[i][1],array[i+1][0],array[i+1][1]) > 0.5 || calcurvature(array[i-1][0],array[i-1][1],array[i][0],array[i][1],array[i+1][0],array[i+1][1]) < -0.5){
				array2.push_back({array[i][0],array[i][1],(calcurvature(array[i-2][0],array[i-2][1],array[i-1][0],array[i-1][1],array[i][0],array[i][1])+calcurvature(array[i][0],array[i][1],array[i+1][0],array[i+1][1],array[i+2][0],array[i+2][1]))/2});
			}
			else{
				array2.push_back({array[i][0],array[i][1],calcurvature(array[i-1][0],array[i-1][1],array[i][0],array[i][1],array[i+1][0],array[i+1][1])});
			}
        }
    }
    return array2;
}

double calslope (double ax, double ay, double bx, double by){
    if(by>=ay){
        return acos((bx-ax)/sqrt(pow(bx-ax,2)+pow(by-ay,2)));
    }
    else{
        return -acos((bx-ax)/sqrt(pow(bx-ax,2)+pow(by-ay,2)));
    }
}

std::vector< std::vector<double> > convert_pcl_to_vector(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  std::vector< std::vector<double> > array;
	for (int h = 0; h < cloud->height; h++) {
		for (int w = 0; w < cloud->width; w++) {
			pcl::PointXYZ &point = cloud->points[w + h * cloud->width];
			array.push_back({point.x, point.y});
			}
	}
  return array;
}

void GetRPY(const geometry_msgs::Quaternion &q,
    double &roll,double &pitch,double &yaw){
  tf2::Quaternion quat(q.x,q.y,q.z,q.w);
  tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
}

double relative_coordinates_horizontal (double x0, double y0, double euler_z, double x, double y){
    // double x_relative = x-x0;
    // double y_relative = y-y0;
    // (x-x0) * cos(-euler_z) - (y-y0) * sin (-euler_z);
    return -((x-x0) * sin(-euler_z) + (y-y0) * cos(-euler_z));
}

std::pair<double, double> control_main(double x, double y, double euler_z, std::vector< std::vector<double> > path){
  int closest_index = closest_point_index(x, y, path);
  std_msgs::Int32 msg;
  msg.data = closest_index;  // Integer value to publish
  pub_position.publish(msg);  // Publish the message
  for (int j = 0; j < point_vk.size(); j++){
    if(j != point_vk.size() - 1){
      if(point_vk[j][0] <= closest_index && closest_index < point_vk[j+1][0]){
      vr0 = point_vk[j][1];
      K2 = point_vk[j][2];
      K3 = point_vk[j][3];
      break;
      }
    }
    else{
      vr0 = point_vk[j][1];
      K2 = point_vk[j][2];
      K3 = point_vk[j][3];
    }
  }
  double path_slope;
  if(closest_index==path.size()-1){
    path_slope = calslope(path[closest_index][0], path[closest_index][1], path[0][0], path[0][1]);
  }
  else{
    path_slope = calslope(path[closest_index][0], path[closest_index][1], path[closest_index+1][0], path[closest_index+1][1]);
  }
  double e2 = relative_coordinates_horizontal(x,y,euler_z,path[closest_index][0], path[closest_index][1]);
  double e3 = euler_z - path_slope;
  std_msgs::Float64MultiArray msg_array;
  msg_array.data.push_back(e2);
  msg_array.data.push_back(e3);
  pub_e.publish(msg_array);
  double curvature = path[closest_index][2];

  double vr = vr0 * (1 - e2 * curvature) / cos(e3);
  double omegar = curvature * vr * cos(e3);

  double v = vr * cos(e3);
  double omega = omegar - vr * (K2 * e2 + K3 * sin(e3));

  if(omega/v > c_max){
    omega = c_max * v;
    }
  else if(omega/v < -c_max){
    omega = -c_max * v;
  }
  //ROS_INFO("e2=%lf e3=%lf curvature=%lf", e2, e3, curvature);
  return std::make_pair(v, omega);
}

void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_in)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_in, *cloud_ptr);
    preliminary_path = add_curvature(convert_pcl_to_vector(cloud_ptr));
    state1 += 1;
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "control");
  ros::NodeHandle nh;
  SetParameters();
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);
  ros::Time::waitForValid();
  ros::WallTime wall_begin;
  ros::Subscriber sbs_pointcloud = nh.subscribe<sensor_msgs::PointCloud2>(TRAJECTORY_TOPIC, 1, pointcloudCallback);
  ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/can_cmd", 1);
  pub_position = nh.advertise<std_msgs::Int32>(POSITION_TOPIC, 1);
  pub_e = nh.advertise<std_msgs::Float64MultiArray>("/horizontal_and_attitude_error", 1);
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
      if ((ros::WallTime::now() - wall_begin).sec < 5.0){
        geometry_msgs::Twist msg_cmd;
        msg_cmd.linear.x = 1.0;
        msg_cmd.angular.z = 0.0;
        pub.publish(msg_cmd);
      }
      else{
        if(state1>=1){
          geometry_msgs::TransformStamped transformStamped;
          transformStamped = tf_buffer.lookupTransform("map", "base_link", ros::Time(0));
          double euler_x, euler_y, euler_z;
          GetRPY(transformStamped.transform.rotation, euler_x, euler_y, euler_z);
          geometry_msgs::Twist msg_cmd;
          double a;
          double b;
          std::tie(a, b) = control_main(transformStamped.transform.translation.x, transformStamped.transform.translation.y, euler_z, preliminary_path);
          msg_cmd.linear.x = a;
          msg_cmd.angular.z = b;
          pub = nh.advertise<geometry_msgs::Twist>("/can_cmd", 1);
          pub.publish(msg_cmd);
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