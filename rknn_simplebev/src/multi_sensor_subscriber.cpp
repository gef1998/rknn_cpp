#include "multi_sensor_subscriber.hpp"
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <random>
#include <algorithm>


MultiSensorSubscriber::MultiSensorSubscriber(ros::NodeHandle& nh, const SensorCallback& callback)
    : nh_(nh), callback_(callback), running_(false), frame_count_(0),
      tf_buffer_(ros::Duration(10.0)), tf_listener_(tf_buffer_),
      front_left_sub_(nh_, "/front/left/image_raw", 5),
    //   front_right_sub_(nh_, "/front/right/image_raw", 5),
      right_left_sub_(nh_, "/right/left/image_raw", 5),
    //   right_right_sub_(nh_, "/right/right/image_raw", 5),
      back_left_sub_(nh_, "/back/left/image_raw", 5),
    //   back_right_sub_(nh_, "/back/right/image_raw", 5),
      left_left_sub_(nh_, "/left/left/image_raw", 5),
    //   left_right_sub_(nh_, "/left/right/image_raw", 5),
      base_frame_("base_footprint")
{
    // 分配输出缓冲区
    image_buffer_ = std::make_unique<unsigned char[]>(TOTAL_IMAGE_SIZE);
    pointcloud_buffer_ = std::make_unique<rknpu2::float16[]>(TOTAL_POINTCLOUD_SIZE);

    // 创建时间同步器 - 只同步8个图像
    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), 
        front_left_sub_, 
        // front_right_sub_, 
        right_left_sub_, 
        // right_right_sub_, 
        back_left_sub_, 
        // back_right_sub_, 
        left_left_sub_
        // left_right_sub_
    );
    
    // 注册图像同步回调
    sync_->registerCallback(boost::bind(&MultiSensorSubscriber::imageCallback, this, 
                                      _1, _2, _3, _4
                                    //   ,_5, _6, _7, _8
                                    ));
    
    // 独立订阅激光数据
    front_laser_sub_ = nh_.subscribe("/scan_emma_nav_front", 5, 
                                    &MultiSensorSubscriber::frontLaserCallback, this);
    back_laser_sub_ = nh_.subscribe("/scan_emma_nav_back", 5, 
                                   &MultiSensorSubscriber::backLaserCallback, this);
    
    ROS_INFO("Multi-sensor subscriber initialized");
    ROS_INFO("Subscribed image topics:");
    ROS_INFO("  /front/left/image_raw");
    // ROS_INFO("  /front/right/image_raw");
    ROS_INFO("  /right/left/image_raw");
    // ROS_INFO("  /right/right/image_raw");
    ROS_INFO("  /back/left/image_raw"); 
    // ROS_INFO("  /back/right/image_raw");
    ROS_INFO("  /left/left/image_raw");
    // ROS_INFO("  /left/right/image_raw");
    ROS_INFO("Subscribed laser topics (independent):");
    ROS_INFO("  /scan_emma_nav_front");
    ROS_INFO("  /scan_emma_nav_back");
    ROS_INFO("Target base frame: %s", base_frame_.c_str());
}

MultiSensorSubscriber::~MultiSensorSubscriber() {
    stop();
    ROS_INFO("Multi-sensor subscriber destroyed");
}

void MultiSensorSubscriber::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;
    ROS_INFO("Started subscribing to multi-sensor data");
}

void MultiSensorSubscriber::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    ROS_INFO("Stopped subscribing to multi-sensor data");
}

void MultiSensorSubscriber::frontLaserCallback(const sensor_msgs::LaserScanConstPtr& laser_msg) {
    std::lock_guard<std::mutex> lock(laser_mutex_);
    latest_front_laser_ = laser_msg;
    ROS_DEBUG("Received front laser data with %zu points", laser_msg->ranges.size());
}

void MultiSensorSubscriber::backLaserCallback(const sensor_msgs::LaserScanConstPtr& laser_msg) {
    std::lock_guard<std::mutex> lock(laser_mutex_);
    latest_back_laser_ = laser_msg;
    ROS_DEBUG("Received back laser data with %zu points", laser_msg->ranges.size());
}

void MultiSensorSubscriber::imageCallback(
    const sensor_msgs::ImageConstPtr& front_left,
    // const sensor_msgs::ImageConstPtr& front_right,
    const sensor_msgs::ImageConstPtr& right_left,
    // const sensor_msgs::ImageConstPtr& right_right,
    const sensor_msgs::ImageConstPtr& back_left,
    // const sensor_msgs::ImageConstPtr& back_right,
    const sensor_msgs::ImageConstPtr& left_left)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_) {
        return;
    }
    
    try {
        // 获取最新的激光点云数据
        getLatestLaserPointCloud(pointcloud_buffer_.get());


        // // 打印latest_front_laser_的时间戳
        // if (latest_front_laser_) {
        //     ROS_INFO("Latest front laser timestamp: %f", latest_front_laser_->header.stamp.toSec());
        // }
        // if (latest_back_laser_) {
        //     ROS_INFO("Latest back laser timestamp: %f", latest_back_laser_->header.stamp.toSec());
        // }
        // // 打印相机数据时间戳
        // ROS_INFO("Front left image timestamp: %f", front_left->header.stamp.toSec()); 
        // // ROS_INFO("Front right image timestamp: %f", front_right->header.stamp.toSec());
        // ROS_INFO("Right left image timestamp: %f", right_left->header.stamp.toSec());
        // // ROS_INFO("Right right image timestamp: %f", right_right->header.stamp.toSec());
        // ROS_INFO("Back left image timestamp: %f", back_left->header.stamp.toSec());
        // // ROS_INFO("Back right image timestamp: %f", back_right->header.stamp.toSec());
        // // ROS_INFO("Left left image timestamp: %f", left_left->header.stamp.toSec());
        // ROS_INFO("Left right image timestamp: %f", left_right->header.stamp.toSec());

        // 处理8个摄像头图像
        std::vector<cv::Mat> images;
        images.reserve(NUM_CAMERAS);
        
        images.push_back(preprocessImage(front_left));
        images.push_back(preprocessImage(right_left));
        images.push_back(preprocessImage(back_left));
        images.push_back(preprocessImage(left_left));

        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));
        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));
        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));
        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));
        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));
        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));
        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));
        // images.push_back(cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3));

        // // resized_image 可视化
        // cv::imshow("Resized Image", images[0]);
        // cv::waitKey(1); // 非阻塞显示，允许实时更新

        // 合并图像
        mergeImages(images, image_buffer_.get());
        
        // 调用回调函数进行推理（npu推理）
        if (callback_) {
            callback_(image_buffer_.get(), pointcloud_buffer_.get());
        }
        
        // 更新统计信息
        frame_count_++;
        
        if (frame_count_ % 60 == 0) {
            ROS_INFO("Processed %d multi-sensor frames", frame_count_);
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Multi-sensor processing error: %s", e.what());
    }
}

void MultiSensorSubscriber::getLatestLaserPointCloud(rknpu2::float16* output_buffer) {
    std::lock_guard<std::mutex> lock(laser_mutex_);
    
    // 初始化输出缓冲区为0
    memset(output_buffer, 0, TOTAL_POINTCLOUD_SIZE * sizeof(rknpu2::float16));
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr back_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 处理前激光数据
    if (latest_front_laser_) {
        front_cloud = laserToPointCloud(latest_front_laser_);
        front_cloud = transformToBaseFrame(front_cloud, latest_front_laser_->header.frame_id);
    } else {
        ROS_WARN_THROTTLE(2.0, "No front laser data available");
    }
    
    // 处理后激光数据
    if (latest_back_laser_) {
        back_cloud = laserToPointCloud(latest_back_laser_);
        back_cloud = transformToBaseFrame(back_cloud, latest_back_laser_->header.frame_id);
    } else {
        ROS_WARN_THROTTLE(2.0, "No back laser data available");
    }
    
    // 合并并采样点云
    mergeAndSamplePointClouds(front_cloud, back_cloud, output_buffer);
}

cv::Mat MultiSensorSubscriber::preprocessImage(const sensor_msgs::ImageConstPtr& img_msg) {
    // 转换ROS图像为OpenCV格式
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::RGB8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    }
    
    cv::Mat image = cv_ptr->image;
    cv::Mat resized_image;
    
    // 调整图像尺寸到 224x400
    if (image.rows != IMAGE_HEIGHT || image.cols != IMAGE_WIDTH) {
        cv::resize(image, resized_image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);
    } else {
        resized_image = image.clone();
    }

    return resized_image;
}

void MultiSensorSubscriber::mergeImages(const std::vector<cv::Mat>& images, unsigned char* output_buffer) {
    if (images.size() != NUM_CAMERAS) {
        ROS_ERROR("Image count mismatch, expected %d, got %zu", NUM_CAMERAS, images.size());
        return;
    }
    
    unsigned char* buffer_ptr = output_buffer;
    
    // 按顺序复制每个摄像头的数据
    for (int camera_idx = 0; camera_idx < NUM_CAMERAS; ++camera_idx) {
        const cv::Mat& img = images[camera_idx];
        
        if (img.rows != IMAGE_HEIGHT || img.cols != IMAGE_WIDTH || img.channels() != IMAGE_CHANNELS) {
            ROS_ERROR("Camera %d image size mismatch: %dx%dx%d, expected: %dx%dx%d", 
                      camera_idx, img.rows, img.cols, img.channels(),
                      IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
            
            // 用零填充
            memset(buffer_ptr, 0, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS);
        } else {
            // 复制图像数据
            if (img.isContinuous()) {
                memcpy(buffer_ptr, img.data, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS);
            } else {
                // 如果不连续，按行复制
                for (int row = 0; row < IMAGE_HEIGHT; ++row) {
                    memcpy(buffer_ptr + row * IMAGE_WIDTH * IMAGE_CHANNELS,
                           img.ptr<unsigned char>(row),
                           IMAGE_WIDTH * IMAGE_CHANNELS);
                }
            }
        }
        
        buffer_ptr += IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;
    }
    
    ROS_DEBUG("Merged %d images successfully, Total size: %d", NUM_CAMERAS, TOTAL_IMAGE_SIZE);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MultiSensorSubscriber::laserToPointCloud(
    const sensor_msgs::LaserScanConstPtr& laser_msg) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 将LaserScan转换为点云
    float angle = laser_msg->angle_min;
    for (size_t i = 0; i < laser_msg->ranges.size(); ++i) {
        float range = laser_msg->ranges[i];
        
        // 检查距离是否有效
        if (range >= laser_msg->range_min && range <= laser_msg->range_max) {
            pcl::PointXYZ point;
            point.x = range * cos(angle);
            point.y = range * sin(angle);
            point.z = 0.0;  // 2D激光，Z坐标为0
            
            cloud->points.push_back(point);
        }
        
        angle += laser_msg->angle_increment;
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    
    ROS_DEBUG("Converted laser scan to point cloud with %zu points", cloud->points.size());
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MultiSensorSubscriber::transformToBaseFrame(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
    const std::string& source_frame) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    try {
        // 获取变换矩阵
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped = tf_buffer_.lookupTransform(base_frame_, source_frame, 
                                                      ros::Time(0), ros::Duration(1.0));
        
        // 转换为PCL变换矩阵
        Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
        
        // 平移
        transform_matrix(0, 3) = transform_stamped.transform.translation.x;
        transform_matrix(1, 3) = transform_stamped.transform.translation.y;
        transform_matrix(2, 3) = transform_stamped.transform.translation.z;
        
        // 旋转（四元数转旋转矩阵）
        tf2::Quaternion q(
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z,
            transform_stamped.transform.rotation.w);
        tf2::Matrix3x3 rotation_matrix(q);
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                transform_matrix(i, j) = rotation_matrix[i][j];
            }
        }
        
        // 应用变换
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform_matrix);
        
        ROS_DEBUG("Transformed point cloud from %s to %s with %zu points", 
                 source_frame.c_str(), base_frame_.c_str(), transformed_cloud->points.size());
        
    } catch (tf2::TransformException& ex) {
        ROS_WARN("Could not transform from %s to %s: %s", 
                source_frame.c_str(), base_frame_.c_str(), ex.what());
        // 如果变换失败，返回原始点云
        *transformed_cloud = *cloud;
    }
    
    return transformed_cloud;
}

void MultiSensorSubscriber::mergeAndSamplePointClouds(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& front_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& back_cloud,
    rknpu2::float16* output_buffer) {
    
    // 合并两个点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    *merged_cloud = *front_cloud;
    *merged_cloud += *back_cloud;
    
    size_t total_points = merged_cloud->points.size();
    
    if (total_points == 0) {
        ROS_WARN_THROTTLE(5.0, "No valid points in merged point cloud");
        return;
    }

    // // 保存合并后的点云到bin文件，方便CC可视化 TEST
    // static int pc_file_counter = 1;
    // char pc_filename[64];
    // snprintf(pc_filename, sizeof(pc_filename), "merged_cloud_%d.bin", pc_file_counter++);
    // FILE* pc_fp = fopen(pc_filename, "wb");
    // if (pc_fp) {
    //     for (size_t i = 0; i < total_points; ++i) {
    //         float xyz[3] = {merged_cloud->points[i].x, merged_cloud->points[i].y, merged_cloud->points[i].z};
    //         fwrite(xyz, sizeof(float), 3, pc_fp);
    //     }
    //     fclose(pc_fp);
    //     ROS_INFO("已保存合并点云到文件: %s, 点数: %zu", pc_filename, total_points);
    // } else {
    //     ROS_WARN("无法保存合并点云到文件: %s", pc_filename);
    // }
    
    // 点数少于或等于3000，直接复制并用0填充剩余部分
    for (size_t i = 0; i < total_points; ++i) {
        output_buffer[i * POINT_DIMS + 0] = (rknpu2::float16)merged_cloud->points[i].x;
        output_buffer[i * POINT_DIMS + 1] = (rknpu2::float16)merged_cloud->points[i].y;
        output_buffer[i * POINT_DIMS + 2] = (rknpu2::float16)merged_cloud->points[i].z;
    }
    ROS_DEBUG("Filled point cloud with %zu points (padded with zeros to %d)", 
                total_points, POINTCLOUD_SIZE);
} 