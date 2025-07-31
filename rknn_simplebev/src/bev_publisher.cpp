#include "bev_publisher.hpp"

BEVPublisher::BEVPublisher(ros::NodeHandle& nh, 
                          const std::string& topic_name,
                          int queue_size)
    : nh_(nh), topic_name_(topic_name), published_count_(0) {
    
    // 创建LaserScan发布器
    laser_pub_ = nh_.advertise<sensor_msgs::LaserScan>(topic_name_, queue_size);
    
    // 初始化默认变换矩阵（单位矩阵）
    base_T_ref_ = Eigen::Matrix4f::Identity();
        
    // 初始化默认LaserScan参数
    frame_id_ = "base_link";
    angle_min_ = -M_PI;
    angle_max_ = M_PI;
    angle_increment_ = M_PI / 180.0; // 1度
    range_min_ = 0.1f;
    range_max_ = 30.0f;
    
    ROS_INFO("BEV发布器已初始化，话题: %s", topic_name_.c_str());
}

void BEVPublisher::setTransformMatrix(const float transform_array[16]) {
    base_T_ref_ = bev_utils::createTransformMatrix(transform_array);
    ROS_INFO("已更新base_T_ref变换矩阵");
}

void BEVPublisher::setTransformMatrix(const Eigen::Matrix4f& transform_matrix) {
    base_T_ref_ = transform_matrix;
    ROS_INFO("已更新base_T_ref变换矩阵");
}

void BEVPublisher::setBEVConfig(const bev_utils::BEVConfig& config) {
    bev_config_ = config;
    ROS_INFO("已更新BEV配置: %dx%d网格", config.grid_width, config.grid_height);
}

void BEVPublisher::setLaserScanParams(const std::string& frame_id,
                                     float angle_min,
                                     float angle_max,
                                     float angle_increment,
                                     float range_min,
                                     float range_max) {
    frame_id_ = frame_id;
    angle_min_ = angle_min;
    angle_max_ = angle_max;
    angle_increment_ = angle_increment;
    range_min_ = range_min;
    range_max_ = range_max;
    
    ROS_INFO("已更新LaserScan参数: frame_id=%s, 角度范围=[%.2f, %.2f], 距离范围=[%.2f, %.2f]",
             frame_id_.c_str(), angle_min_, angle_max_, range_min_, range_max_);
}

void BEVPublisher::publishBEVResult(const rknpu2::float16* bev_result) {  
    try {
        // 转换BEV结果为LaserScan
        sensor_msgs::LaserScan scan = bev_utils::pointCloudToLaserScan(
            bev_utils::transformPointCloud(
                bev_utils::bevGridToPointCloud(bev_result, bev_config_),
                base_T_ref_
            ),
            frame_id_,
            angle_min_,
            angle_max_,
            angle_increment_,
            range_min_,
            range_max_
        );
        
        // 设置时间戳
        scan.header.stamp = ros::Time::now();
        
        // 发布LaserScan消息
        laser_pub_.publish(scan);
        
        // 更新统计信息
        published_count_++;
        last_publish_time_ = scan.header.stamp;
        
        ROS_DEBUG("已发布BEV LaserScan消息 #%d", published_count_);
        
    } catch (const std::exception& e) {
        ROS_ERROR("发布BEV结果时出错: %s", e.what());
    }
}

void BEVPublisher::printStats() const {
    ROS_INFO("BEV发布器统计信息:");
    ROS_INFO("  话题: %s", topic_name_.c_str());
    ROS_INFO("  已发布消息数: %d", published_count_);
    ROS_INFO("  最后发布时间: %f", last_publish_time_.toSec());
    ROS_INFO("  当前订阅者数量: %d", laser_pub_.getNumSubscribers());
} 