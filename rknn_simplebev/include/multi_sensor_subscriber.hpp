#ifndef MULTI_SENSOR_SUBSCRIBER_HPP
#define MULTI_SENSOR_SUBSCRIBER_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <mutex>
#include <vector>
#include <functional>
#include "fp16/Float16.h"

class MultiSensorSubscriber {
public:
    // 回调函数类型定义 - 现在包含图像数据和点云数据
    using SensorCallback = std::function<void(unsigned char*, rknpu2::float16*)>;
    
    // 构造函数
    MultiSensorSubscriber(ros::NodeHandle& nh, const SensorCallback& callback);
    
    // 析构函数
    ~MultiSensorSubscriber();
    
    // 启动订阅
    void start();
    
    // 停止订阅
    void stop();
    
    // 获取合并后的数据尺寸信息
    static constexpr int NUM_CAMERAS = 4;
    static constexpr int IMAGE_HEIGHT = 224;
    static constexpr int IMAGE_WIDTH = 400;
    static constexpr int IMAGE_CHANNELS = 3;
    static constexpr int TOTAL_IMAGE_SIZE = NUM_CAMERAS * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;
    
    // 点云数据常量
    static constexpr int POINTCLOUD_SIZE = 3000;  // 3000个点
    static constexpr int POINT_DIMS = 3;  // x, y, z
    static constexpr int TOTAL_POINTCLOUD_SIZE = POINTCLOUD_SIZE * POINT_DIMS;

private:
    // 8个摄像头同步回调函数
    void imageCallback(
        const sensor_msgs::ImageConstPtr& front_left,
        // const sensor_msgs::ImageConstPtr& front_right,
        const sensor_msgs::ImageConstPtr& right_left,
        // const sensor_msgs::ImageConstPtr& right_right,
        const sensor_msgs::ImageConstPtr& back_left,
        // const sensor_msgs::ImageConstPtr& back_right,
        const sensor_msgs::ImageConstPtr& left_left
        // const sensor_msgs::ImageConstPtr& left_right
    );
    
    // 激光数据回调函数（独立处理）
    void frontLaserCallback(const sensor_msgs::LaserScanConstPtr& laser_msg);
    void backLaserCallback(const sensor_msgs::LaserScanConstPtr& laser_msg);
    
    // 单个图像预处理函数
    cv::Mat preprocessImage(const sensor_msgs::ImageConstPtr& img_msg);
    
    // 合并8个图像
    void mergeImages(const std::vector<cv::Mat>& images, unsigned char* output_buffer);
    
    // 激光数据转换为点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserToPointCloud(
        const sensor_msgs::LaserScanConstPtr& laser_msg);
    
    // 将点云转换到base坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformToBaseFrame(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const std::string& source_frame);
    
    // 合并两个点云并采样到3000个点
    void mergeAndSamplePointClouds(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& front_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& back_cloud,
        rknpu2::float16* output_buffer);
    
    // 获取最新的激光点云数据
    void getLatestLaserPointCloud(rknpu2::float16* output_buffer);
    
    ros::NodeHandle& nh_;
    SensorCallback callback_;
    
    // TF监听器
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // 图像消息订阅器
    message_filters::Subscriber<sensor_msgs::Image> front_left_sub_;
    // message_filters::Subscriber<sensor_msgs::Image> front_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> right_left_sub_;
    // message_filters::Subscriber<sensor_msgs::Image> right_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> back_left_sub_;
    // message_filters::Subscriber<sensor_msgs::Image> back_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> left_left_sub_;
    // message_filters::Subscriber<sensor_msgs::Image> left_right_sub_;
    
    // 激光消息订阅器（独立订阅）
    ros::Subscriber front_laser_sub_;
    ros::Subscriber back_laser_sub_;
    
    // 时间同步器 - 只同步8个图像
    typedef message_filters::sync_policies::ApproximateTime<
        // sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
        sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image
    > SyncPolicy;
    
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    // 输出缓冲区
    std::unique_ptr<unsigned char[]> image_buffer_;
    std::unique_ptr<rknpu2::float16[]> pointcloud_buffer_;
    
    // 激光数据缓存
    sensor_msgs::LaserScanConstPtr latest_front_laser_;
    sensor_msgs::LaserScanConstPtr latest_back_laser_;
    std::mutex laser_mutex_;
    
    // 互斥锁
    std::mutex mutex_;
    
    // 运行状态
    bool running_;
    
    // 统计信息
    int frame_count_;
    ros::Time last_fps_time_;
    
    // 基础坐标系名称
    std::string base_frame_;
};

#endif // MULTI_SENSOR_SUBSCRIBER_HPP 