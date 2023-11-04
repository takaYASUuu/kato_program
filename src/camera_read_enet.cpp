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

#include <chrono>

// caffe
#include "classifier_enet.h"
std::string model_path = "/home/nvidia/catkin_ws/src/kato_program/model/bn_conv_merged_model_5label2.prototxt";
std::string weights_path = "/home/nvidia/catkin_ws/src/kato_program/model/bn_conv_merged_weights_5label2.caffemodel";
std::string lut_path;

int flg_show_images;
int flg_count_time;
int flg_compress;
int compress_quality;
int camera_number;

std::string IMAGE_TOPIC;
double ORIGINAL_WIDTH_VALUE;
double ORIGINAL_HEIGHT_VALUE;
double RESIZED_WIDTH_VALUE;
double RESIZED_HEIGHT_VALUE;

cv::Mat enet_result;

// caffe initialize
Classifier classifier(model_path, weights_path);

void SetParameters(){
  ros::NodeHandle nh;
  nh.getParam("camera_image_imput", IMAGE_TOPIC);
  nh.getParam("resized_width", RESIZED_WIDTH_VALUE);
  nh.getParam("resized_height", RESIZED_HEIGHT_VALUE);
  nh.getParam("original_width", ORIGINAL_WIDTH_VALUE);
  nh.getParam("original_height", ORIGINAL_HEIGHT_VALUE);
  nh.getParam("show_image", flg_show_images);
  nh.getParam("count_time", flg_count_time);
  nh.getParam("camera_number", camera_number);
  nh.getParam("caffe_lut_path", lut_path);
}


int main (int argc, char** argv)
{
  // Initialize the ROS Node "ros(cpp_pcl_example"
  ros::init (argc, argv, "camera_read");
  ros::NodeHandle nh;
  SetParameters();
  image_transport::ImageTransport it(nh);
  
  // publish image
  image_transport::Publisher image_pub1;
  image_pub1 = it.advertise(IMAGE_TOPIC, 1);

  cv::VideoCapture cap1;
  cap1.open(camera_number, cv::CAP_V4L2);
  //cap1.open(camera_number);
  if (!cap1.isOpened()) // カメラデバイスが正常にオープンしたか確認．
  {
      // 読み込みに失敗したときの処理
      cout << "camera disconnect" << endl;
      return -1;
  }
  cap1.set(cv::CAP_PROP_FRAME_WIDTH, ORIGINAL_WIDTH_VALUE);
  cap1.set(cv::CAP_PROP_FRAME_HEIGHT, ORIGINAL_HEIGHT_VALUE);
  cap1.set(cv::CAP_PROP_BUFFERSIZE, 1); // buffer_size 小さい方が遅延しない

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    cv::Mat image1;
    std::chrono::system_clock::time_point start1, end1;
    
    try{
      //start1 = std::chrono::system_clock::now(); // 計測開始時間
      cap1.read(image1);
      // enet
      caffe::Blob<float> *output_layer;
      output_layer = classifier.Predict(image1);
      Mat enet_output = classifier.Visualization2(output_layer);
      //resize(image1, image1, cv::Size(), RESIZED_WIDTH_VALUE/image1.cols ,RESIZED_HEIGHT_VALUE/image1.rows);
      enet_result = enet_output;
      if (flg_show_images)
      {
        cv::cvtColor(enet_output.clone(), enet_output, cv::COLOR_GRAY2BGR);
        cv::Mat label_colours = cv::imread(lut_path, 1);
        cv::cvtColor(label_colours, label_colours, cv::COLOR_RGB2BGR);
        cv::Mat output_image;
        cv::LUT(enet_output, label_colours, output_image);
        cv::Mat result = overlay(image1, output_image);
        cv::imshow("image", result);
        cv::waitKey(1);
      }
      //end1 = std::chrono::system_clock::now();   // 計測終了時間
      //double elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count(); // 処理に要した時間をミリ秒に変換
      //if(flg_count_time){cout << "get image:" << elapsed1 << endl;}
        sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image1).toImageMsg();
        image_pub1.publish(msg1);
    }
    catch (cv::Exception &e)
    {
      cout << "error" << endl;
      continue;
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
