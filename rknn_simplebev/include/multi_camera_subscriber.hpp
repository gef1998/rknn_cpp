#ifndef MULTI_CAMERA_SUBSCRIBER_HPP
#define MULTI_CAMERA_SUBSCRIBER_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <mutex>
#include <vector>
#include <functional>

class MultiCameraSubscriber {
public:
    // 回调函数类型定义
    using ImageCallback = std::function<void(unsigned char*)>;
    
    // 构造函数
    MultiCameraSubscriber(ros::NodeHandle& nh, const ImageCallback& callback);
    
    // 析构函数
    ~MultiCameraSubscriber();
    
    // 启动订阅
    void start();
    
    // 停止订阅
    void stop();
    
    // 获取合并后的图像尺寸信息
    static constexpr int NUM_CAMERAS = 8;
    static constexpr int IMAGE_HEIGHT = 224;
    static constexpr int IMAGE_WIDTH = 400;
    static constexpr int IMAGE_CHANNELS = 3;
    static constexpr int TOTAL_SIZE = NUM_CAMERAS * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;

private:
    // 8个摄像头同步回调函数
    void imageCallback(
        const sensor_msgs::ImageConstPtr& front_left,
        const sensor_msgs::ImageConstPtr& front_right,
        const sensor_msgs::ImageConstPtr& right_left,
        const sensor_msgs::ImageConstPtr& right_right,
        const sensor_msgs::ImageConstPtr& back_left,
        const sensor_msgs::ImageConstPtr& back_right,
        const sensor_msgs::ImageConstPtr& left_left,
        const sensor_msgs::ImageConstPtr& left_right);
    
    // 单个图像预处理函数
    cv::Mat preprocessImage(const sensor_msgs::ImageConstPtr& img_msg);
    
    // 合并8个图像
    void mergeImages(const std::vector<cv::Mat>& images, unsigned char* output_buffer);
    
    ros::NodeHandle& nh_;
    ImageCallback callback_;
    
    // 消息订阅器
    message_filters::Subscriber<sensor_msgs::Image> back_left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> back_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> front_left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> front_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> left_left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> left_right_sub_;
    message_filters::Subscriber<sensor_msgs::Image> right_left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> right_right_sub_;
    
    // 时间同步器
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
        sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image
    > SyncPolicy;
    
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    // 输出缓冲区
    std::unique_ptr<unsigned char[]> output_buffer_;
    
    // 互斥锁
    std::mutex mutex_;
    
    // 运行状态
    bool running_;
    
    // 统计信息
    int frame_count_;
    ros::Time last_fps_time_;
};

#endif // MULTI_CAMERA_SUBSCRIBER_HPP 