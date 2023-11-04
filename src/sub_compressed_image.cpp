#include <iostream>
#include <sstream> // stringstream
using namespace std;

#include <opencv2/opencv.hpp> // Include OpenCV API

// ROS
#include <ros/ros.h>
#include <std_msgs/String.h> // image_name
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CompressedImage.h>

#include <chrono>

int flg_show_compressed_image;
int flg_count_time;

std::string IMAGE_TOPIC_COMPRESSED;

void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("camera_image_imput_compress", IMAGE_TOPIC_COMPRESSED);
  nh.getParam("show_compressed_image", flg_show_compressed_image);
  nh.getParam("count_time", flg_count_time);
}

void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
{
  try
  {
    // 圧縮されたイメージをcv::Mat形式に変換
    cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

    // ウィンドウに表示
    cv::imshow("image", img);
    cv::waitKey(1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

int main (int argc, char** argv)
{
  // Initialize the ROS Node "ros(cpp_pcl_example"
  ros::init (argc, argv, "sub_compressed_image");
  ros::NodeHandle nh;
  SetParameters();
  image_transport::ImageTransport it(nh);
  ros::Subscriber sub = nh.subscribe<sensor_msgs::CompressedImage>(IMAGE_TOPIC_COMPRESSED, 1, imageCallback);
  //image_transport::Subscriber sub = it.subscribe(IMAGE_TOPIC_COMPRESSED, 1, imageCallback);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
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
