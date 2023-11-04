//
// Created by atsumi on 20/02/06.
//


#include "classifier_enet.h"

using namespace std;
using namespace caffe;

Classifier::Classifier(){
  return;
}
Classifier::Classifier(const string& model_file,
                       const string& trained_file) {

#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);  
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  return;
}


//caffe::Blob<float>* Classifier::Predict(const cv::Mat& img, std::string LUT_file) {
caffe::Blob<float>* Classifier::Predict(const cv::Mat& img) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
		       input_geometry_.height, input_geometry_.width);

  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();
  // net_->ForwardPrefilled();


  /* Copy the output layer to a std::vector */
  caffe::Blob<float>* output_layer = net_->output_blobs()[0];

  //Visualization(output_layer, LUT_file);

  return output_layer;
}


cv::Mat Classifier::Visualization(caffe::Blob<float>* output_layer,
                                  std::string LUT_file) {

  //    cv::Mat merged_output_image = cv::Mat(output_layer->height(),
  //                                          output_layer->width(), CV_32F,
  //                                          const_cast<float *>(output_layer->cpu_data()));

  int width = output_layer->width();
  int height = output_layer->height();
  int channels = output_layer->channels();
  int num = output_layer->num();

  //std::cout << "output_blob(n,c,h,w) = " << num << ", " << channels << ", "
	//    << height << ", " << width << std::endl;

  // compute argmax
  cv::Mat class_each_row (channels, width*height, CV_32FC1, const_cast<float *>(output_layer->cpu_data()));
  class_each_row = class_each_row.t(); // transpose to make each row with all probabilities
  cv::Point maxId;    // point [x,y] values for index of max
  double maxValue;    // the holy max value itself
  cv::Mat prediction_map(height, width, CV_8UC1);
  for (int i=0;i<class_each_row.rows;i++){
    minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);
    prediction_map.at<uchar>(i) = maxId.x;
  }

  cv::cvtColor(prediction_map.clone(), prediction_map, cv::COLOR_GRAY2BGR);
  cv::Mat label_colours = cv::imread(LUT_file,1);
  cv::cvtColor(label_colours, label_colours, cv::COLOR_RGB2BGR);
  cv::Mat output_image;
  LUT(prediction_map, label_colours, output_image);


  //cv::imshow( "segmentation", output_image);

  return output_image;
}

cv::Mat Classifier::Visualization2(caffe::Blob<float>* output_layer) {

  //    cv::Mat merged_output_image = cv::Mat(output_layer->height(),
  //                                          output_layer->width(), CV_32F,
  //                                          const_cast<float *>(output_layer->cpu_data()));

  int width = output_layer->width();
  int height = output_layer->height();
  int channels = output_layer->channels();
  int num = output_layer->num();

  //std::cout << "output_blob(n,c,h,w) = " << num << ", " << channels << ", "
	//    << height << ", " << width << std::endl;

  // compute argmax
  cv::Mat class_each_row (channels, width*height, CV_32FC1, const_cast<float *>(output_layer->cpu_data()));
  class_each_row = class_each_row.t(); // transpose to make each row with all probabilities
  cv::Point maxId;    // point [x,y] values for index of max
  double maxValue;    // the holy max value itself
  cv::Mat prediction_map(height, width, CV_8UC1);
  for (int i=0;i<class_each_row.rows;i++){
    minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);
    prediction_map.at<uchar>(i) = maxId.x;
  }

  return prediction_map;
}


void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {

  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();

  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }

  return;
}



void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {

  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /*
    This operation will write the separate BGR planes directly to the
    input layer of the network because it is wrapped by the cv::Mat
    objects in input_channels.
  */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
	== net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

  return;
}


cv::Mat overlay(cv::Mat src, cv::Mat src2, double alpha){
  cv::Mat ret(src.size(), CV_8UC3, cv::Scalar::all(0));
  resize(src2, src2, src.size());
  for (int i = 0; i < src.cols; i++)
  {
    for (int j = 0; j < src.rows; j++)
    {
      for (int c = 0; c < 3; c++)
      {
        double val1 = src.data[3 * (src.cols * j + i) + c];
        double val2 = src2.data[3 * (src2.cols * j + i) + c];
        ret.data[3 * (ret.cols * j + i) + c] = val1 + (val2 - val1) * alpha;
      }
    }
  }
  return ret;
}