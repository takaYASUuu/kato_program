#ifndef CAMERA_INFO_HPP
#define CAMERA_INFO_HPP

class camera_param
{
public:
  int width;
  int height;
  double fx;
  double fy;
  double cx;
  double cy;
  double newfx;
  double newfy;
  double newcx;
  double newcy;
  cv::Mat camera_matrix;
  cv::Mat extrinsic_matrix;
  cv::Mat rotation;
  cv::Mat translation;
  cv::Mat distortion;
  Eigen::Matrix<double, 3, 3> rotation_mat;
  Eigen::Vector3d translation_vec;
  cv::Size2i image_size;
  cv::Mat newCameraMatrix;
  cv::Mat map1;
  cv::Mat map2;
  // void set(cv::Mat extrinsic_matrix, cv::Mat camera_matrix, cv::Mat distortion, cv::Size2i image_size);
  void set(cv::Mat extrinsic_matrix, cv::Mat camera_matrix, cv::Mat new_camera_matrix, cv::Mat distortion, cv::Size2i image_size);
  bool read_yaml(std::string file_name);
  void offset(double offset_x, double offset_y, double offset_z, double offset_x_angle, double offset_y_angle, double offset_z_angle);
  void print();
  cv::Mat undistortImage(cv::Mat image);
  // cv::Mat undistortPoint(int x, int y);
};

void camera_param::set(cv::Mat extrinsic_matrix, cv::Mat camera_matrix, cv::Mat new_camera_matrix, cv::Mat distortion, cv::Size2i image_size)
{
  this->extrinsic_matrix = extrinsic_matrix;
  this->camera_matrix = camera_matrix;
  this->distortion = distortion;
  this->image_size = image_size;
  width = image_size.width;
  height = image_size.height;
  fx = camera_matrix.at<double>(0, 0);
  fy = camera_matrix.at<double>(1, 1);
  cx = camera_matrix.at<double>(0, 2);
  cy = camera_matrix.at<double>(1, 2);
  rotation = extrinsic_matrix(cv::Rect(0, 0, 3, 3));
  translation = extrinsic_matrix(cv::Rect(3, 0, 1, 3));
  cv::cv2eigen(rotation, rotation_mat);
  cv::cv2eigen(translation, translation_vec);
  newCameraMatrix = new_camera_matrix;
  cv::initUndistortRectifyMap(camera_matrix, distortion, cv::Mat(), newCameraMatrix, image_size, CV_32FC1, map1, map2);
  newfx = newCameraMatrix.at<double>(0, 0);
  newfy = newCameraMatrix.at<double>(1, 1);
  newcx = newCameraMatrix.at<double>(0, 2);
  newcy = newCameraMatrix.at<double>(1, 2);
  std::cout << "camera parameter loaded" << std::endl;
}

// void camera_param::set(cv::Mat extrinsic_matrix, cv::Mat camera_matrix, cv::Mat distortion, cv::Size2i image_size)
// {
//   this->extrinsic_matrix = extrinsic_matrix;
//   this->camera_matrix = camera_matrix;
//   this->distortion = distortion;
//   this->image_size = image_size;
//   width = image_size.width;
//   height = image_size.height;
//   fx = camera_matrix.at<double>(0, 0);
//   fy = camera_matrix.at<double>(1, 1);
//   cx = camera_matrix.at<double>(0, 2);
//   cy = camera_matrix.at<double>(1, 2);
//   rotation = extrinsic_matrix(cv::Rect(0, 0, 3, 3));
//   translation = extrinsic_matrix(cv::Rect(3, 0, 1, 3));
//   cv::cv2eigen(rotation, rotation_mat);
//   cv::cv2eigen(translation, translation_vec);
//   newCameraMatrix = cv::getOptimalNewCameraMatrix(camera_matrix, distortion, cv::Size(1920, 1020), 1.0);
//   cout << newCameraMatrix << endl;
//   // newCameraMatrix = cv::getOptimalNewCameraMatrix(camera_matrix, distortion, cv::Size(1920, 1208), 1.0);
//   // cout << newCameraMatrix << endl;
//   cv::initUndistortRectifyMap(camera_matrix, distortion, cv::Mat(), newCameraMatrix, image_size, CV_32FC1, map1, map2);
//   newfx = newCameraMatrix.at<double>(0, 0);
//   newfy = newCameraMatrix.at<double>(1, 1);
//   newcx = newCameraMatrix.at<double>(0, 2);
//   newcy = newCameraMatrix.at<double>(1, 2);
//   std::cout << "camera parameter loaded" << std::endl;
// }

bool camera_param::read_yaml(std::string file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  if (!fs.isOpened())
  {
    std::cout << "File can not be opened." << std::endl;
    return false;
  }
  try
  {
    fs["CameraExtrinsicMat"] >> extrinsic_matrix;
    fs["CameraMat"] >> camera_matrix;
    fs["DistCoeff"] >> distortion;
    fs["ImageSize"] >> image_size;
    std::cout << "read calibration data" << std::endl;
  }
  catch (...)
  {
    std::cout << "cannot read calibration data" << std::endl;
    return false;
  }
  width = image_size.width;
  height = image_size.height;
  fx = camera_matrix.at<double>(0, 0);
  fy = camera_matrix.at<double>(1, 1);
  cx = camera_matrix.at<double>(0, 2);
  cy = camera_matrix.at<double>(1, 2);
  rotation = extrinsic_matrix(cv::Rect(0, 0, 3, 3));
  translation = extrinsic_matrix(cv::Rect(3, 0, 1, 3));
  newCameraMatrix = cv::getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, 0);
  cv::initUndistortRectifyMap(camera_matrix, distortion, cv::Mat(), newCameraMatrix, image_size, CV_32FC1, map1, map2);
  newfx = newCameraMatrix.at<double>(0, 0);
  newfy = newCameraMatrix.at<double>(1, 1);
  newcx = newCameraMatrix.at<double>(0, 2);
  newcy = newCameraMatrix.at<double>(1, 2);
  return true;
}

void camera_param::offset(double offset_x, double offset_y, double offset_z, double offset_x_angle, double offset_y_angle, double offset_z_angle)
{
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
  rotation = rotation * rot_x.t() * rot_y.t() * rot_z.t();
  translation = translation - rotation * vector;
  cv::cv2eigen(rotation, rotation_mat);
  cv::cv2eigen(translation, translation_vec);
}

void camera_param::print()
{
  // std::cout << "image_size :" << std::endl << image_size << std::endl;
  std::cout << "extrinsic matrix :" << std::endl
            << extrinsic_matrix << std::endl;
  std::cout << "camera matrix :" << std::endl
            << camera_matrix << std::endl;
  std::cout << "distortion :" << std::endl
            << distortion << std::endl;
  std::cout << "new camera matrix :" << std::endl
            << newCameraMatrix << std::endl;
  // std::cout << "rotation matrix :" << std::endl << rotation << std::endl;
  // std::cout << "translation :" << std::endl << translation << std::endl;
  // std::cout << "fx = " << fx << std::endl;
  // std::cout << "fy = " << fy << std::endl;
  // std::cout << "cx = " << cx << std::endl;
  // std::cout << "cy = " << cy << std::endl;
}

cv::Mat camera_param::undistortImage(cv::Mat image)
{
  cv::Mat distorted;
  cv::remap(image, distorted, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
  return distorted;
}

/*
cv::Mat camera_param::undistortPoint(int x, int y){
  double x_new = map1.at<double>(x,y);
  double y_new = map2.at<double>(x,y);
  return (cv::Mat_<double>(2,1)<<x_new,y_new);
}
*/

#endif // CAMERA_INFO_HPP
