//
// Created by atsumi on 20/02/06.
// Modified by oishi
//

//#ifndef CPU_ONLY
//#define CPU_ONLY
//#endif


#define USE_OPENCV 1
#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#ifdef OPENCV4
#include <opencv2/highgui/highgui_c.hpp>
#else
#include <opencv2/highgui.hpp>
#endif
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include <chrono> //Just for time measurement. This library requires compiler and library support for the ISO C++ 2011 standard. This support is currently experimental in Caffe, and must be enabled with the -std=c++11 or -std=gnu++11 compiler options.
#include <string>

#ifdef USE_OPENCV

class Classifier {
public:
    Classifier();
    Classifier(const std::string& model_file,
               const std::string& trained_file);


    //caffe::Blob<float>* Predict(const cv::Mat& img, std::string LUT_file);
    caffe::Blob<float>* Predict(const cv::Mat& img);
    cv::Mat Visualization(caffe::Blob<float>* prediction_map, std::string LUT_file);
    cv::Mat Visualization2(caffe::Blob<float>* prediction_map);

private:
    void SetMean(const std::string& mean_file);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);



private:
    std::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;

};

cv::Mat overlay(cv::Mat src, cv::Mat src2, double alpha = 0.2);

#endif
