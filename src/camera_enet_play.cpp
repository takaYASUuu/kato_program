#include <iostream>
#include <sstream> // stringstream
using namespace std;

#include <opencv2/opencv.hpp> // Include OpenCV API
using namespace cv;

// ROS
#include <ros/ros.h>
#include <std_msgs/String.h> // image_name
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

// caffe
#include "classifier_enet.h"
std::string model_path = "/home/nvidia/catkin_ws/src/kato_program/model/bn_conv_merged_model_5label2.prototxt";
std::string weights_path = "/home/nvidia/catkin_ws/src/kato_program/model/bn_conv_merged_weights_5label2.caffemodel";
std::string lut_path;

int flg_show_images;

std::string IMAGE_TOPIC;
double RESIZED_WIDTH_VALUE;
double RESIZED_HEIGHT_VALUE;
cv::Mat image;

// global
cv::Mat enet_result;

// caffe initialize
Classifier classifier(model_path, weights_path);

void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("camera_image_imput", IMAGE_TOPIC);
  nh.getParam("resized_width", RESIZED_WIDTH_VALUE);
  nh.getParam("resized_height", RESIZED_HEIGHT_VALUE);
  //nh.getParam("caffe_model_path", model_path);
  //nh.getParam("caffe_weights_path", weights_path);
  nh.getParam("caffe_lut_path", lut_path);
  nh.getParam("show_image", flg_show_images);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  try {
    image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    }
    catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      }
  // resize(image, image, cv::Size(), 0.1, 0.1);

  // // 画像の幅を表示する
  // std::cout << "width: " << image.cols << std::endl;
  // // 画像の高さを表示する
  // std::cout << "height: " << image.rows << std::endl;

  //サイズを512 x 256に変更
  resize(image, image, cv::Size(), RESIZED_WIDTH_VALUE/image.cols ,RESIZED_HEIGHT_VALUE/image.rows);

  // // 画像の幅を表示する
  // std::cout << "width: " << image.cols << std::endl;
  // // 画像の高さを表示する
  // std::cout << "height: " << image.rows << std::endl;

  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now(); // 計測開始時間
  // enet
  caffe::Blob<float> *output_layer;
  output_layer = classifier.Predict(image);
  Mat enet_output = classifier.Visualization2(output_layer);
  end = std::chrono::system_clock::now();                                                      // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); // 処理に要した時間をミリ秒に変換
  //std::cout << elapsed << " ms" << std::endl; #時間表示

  enet_result = enet_output;

  std::cout << "height: " << enet_result << std::endl;


  if (flg_show_images)
  {
    cv::cvtColor(enet_output.clone(), enet_output, cv::COLOR_GRAY2BGR);
    cv::Mat label_colours = cv::imread(lut_path, 1);
    cv::cvtColor(label_colours, label_colours, cv::COLOR_RGB2BGR);
    cv::Mat output_image;
    cv::LUT(enet_output, label_colours, output_image);
    cv::Mat result = overlay(image, output_image);
    // show image

    // const string window_name = "enet_result";
    // namedWindow(window_name, WINDOW_NORMAL);
    //imshow(window_name, result);
    // imshow(window_name, output_image);
    //char key = waitKey(1);
    cv::imshow("image", result);
    cv::waitKey(1);
    }
}

int main (int argc, char** argv)
{
  // Initialize the ROS Node "ros(cpp_pcl_example"
  ros::init (argc, argv, "camera_enet_play");
  ros::NodeHandle nh;
  SetParameters();
  // Create a ROS Subscriber to IMAGE_TOPIC with a queue_size of 1 and a callback function to cloud_cb
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber image_sub = it.subscribe(IMAGE_TOPIC, 1, imageCallback);
  // image_transport::Subscriber image_sub = it.subscribe("/gmsl/image_0", 1, imageCallback);
  ros::Rate loop_rate(30);
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