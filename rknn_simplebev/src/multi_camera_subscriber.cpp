#include "multi_camera_subscriber.hpp"
#include <ros/ros.h>

MultiCameraSubscriber::MultiCameraSubscriber(ros::NodeHandle& nh, const ImageCallback& callback)
    : nh_(nh), callback_(callback), running_(false), frame_count_(0),
        front_left_sub_(nh_, "/front/left/image_raw", 5),
        front_right_sub_(nh_, "/front/right/image_raw", 5),
        right_left_sub_(nh_, "/right/left/image_raw", 5),
        right_right_sub_(nh_, "/right/right/image_raw", 5),
        back_left_sub_(nh_, "/back/left/image_raw", 5),
        back_right_sub_(nh_, "/back/right/image_raw", 5),
        left_left_sub_(nh_, "/left/left/image_raw", 5),
        left_right_sub_(nh_, "/left/right/image_raw", 5)
{
    // 分配输出缓冲区
    output_buffer_ = std::make_unique<unsigned char[]>(TOTAL_SIZE);
    
    // 创建时间同步器
    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), 
        back_left_sub_, back_right_sub_, front_left_sub_, front_right_sub_,
        left_left_sub_, left_right_sub_, right_left_sub_, right_right_sub_
    );
    
    // 注册同步回调
    sync_->registerCallback(boost::bind(&MultiCameraSubscriber::imageCallback, this, _1, _2, _3, _4, _5, _6, _7, _8));
    
    
    ROS_INFO("Multi-camera subscriber initialized");
    ROS_INFO("Subscribed topics:");
    ROS_INFO("  /front/left/image_raw");
    ROS_INFO("  /front/right/image_raw");
    ROS_INFO("  /right/left/image_raw");
    ROS_INFO("  /right/right/image_raw");
    ROS_INFO("  /back/left/image_raw"); 
    ROS_INFO("  /back/right/image_raw");
    ROS_INFO("  /left/left/image_raw");
    ROS_INFO("  /left/right/image_raw");
}

MultiCameraSubscriber::~MultiCameraSubscriber() {
    stop();
    ROS_INFO("Multi-camera subscriber destroyed");
}

void MultiCameraSubscriber::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;
    ROS_INFO("Started subscribing to multi-camera images");
}

void MultiCameraSubscriber::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    ROS_INFO("Stopped subscribing to multi-camera images");
}

void MultiCameraSubscriber::imageCallback(
    const sensor_msgs::ImageConstPtr& front_left,
    const sensor_msgs::ImageConstPtr& front_right,
    const sensor_msgs::ImageConstPtr& right_left,
    const sensor_msgs::ImageConstPtr& right_right,
    const sensor_msgs::ImageConstPtr& back_left,
    const sensor_msgs::ImageConstPtr& back_right,
    const sensor_msgs::ImageConstPtr& left_left,
    const sensor_msgs::ImageConstPtr& left_right)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_) {
        return;
    }
    
    try {
        // 预处理8个图像
        std::vector<cv::Mat> images;
        images.reserve(NUM_CAMERAS);
        
        images.push_back(preprocessImage(front_left));
        images.push_back(preprocessImage(front_right));
        images.push_back(preprocessImage(right_left));
        images.push_back(preprocessImage(right_right));
        images.push_back(preprocessImage(back_left));
        images.push_back(preprocessImage(back_right));
        images.push_back(preprocessImage(left_left));
        images.push_back(preprocessImage(left_right));
        // 合并图像
        mergeImages(images, output_buffer_.get());
        // 调用回调函数进行推理（npu推理）
        if (callback_) {
            callback_(output_buffer_.get());
        }
        // 更新统计信息
        frame_count_++;
        
        
    } catch (const std::exception& e) {
        ROS_ERROR("Image processing error: %s", e.what());
    }
}

cv::Mat MultiCameraSubscriber::preprocessImage(const sensor_msgs::ImageConstPtr& img_msg) {
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
    // // 可视化resized_image
    // cv::imshow("Resized Image", resized_image);
    // cv::waitKey(0); // 非阻塞显示，允许实时更新
    return resized_image;
}

void MultiCameraSubscriber::mergeImages(const std::vector<cv::Mat>& images, unsigned char* output_buffer) {
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
    
    ROS_DEBUG("Merge %d image successfully, Total size: %d", NUM_CAMERAS, TOTAL_SIZE);
} 