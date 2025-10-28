#include "bev_publisher.hpp"

BEVPublisher::BEVPublisher(ros::NodeHandle& nh, 
                           int fps,
                           const std::string& topic_name,
                           int queue_size)
    : nh_(nh), topic_name_(topic_name), fps(fps), published_count_(0), tf_buffer_(ros::Duration(10.0)), tf_listener_(tf_buffer_) {
    
    // 创建LaserScan发布器
    laser_pub_ = nh_.advertise<sensor_msgs::LaserScan>(topic_name_, queue_size);
    center_points_pub_ = nh.advertise<sensor_msgs::PointCloud2>("bev_center_points", 10);
    stracks_pub_ = nh.advertise<emma_safe_msgs::PersonStateArray>("person_states", 10);
    writer = cv::VideoWriter("tracker_demo.avi",cv::VideoWriter::fourcc('X','V','I','D'), fps, cv::Size(960, 960));
    tracker = BYTETracker(fps, 30);

    // 初始化默认变换矩阵（单位矩阵）
    base_T_mem_ = Eigen::Matrix4f::Identity();
        
    // 初始化默认LaserScan参数
    frame_id_ = "base_footprint";
    angle_min_ = -M_PI;
    angle_max_ = M_PI;
    angle_increment_ = M_PI / 180.0 / 4; // 0.5度
    range_min_ = 0.0500000007451f; // 5cm
    range_max_ = 30.0f;
    
    ROS_INFO("BEV publisher is initialized, Topic: %s", topic_name_.c_str());
}

void BEVPublisher::setTransformMatrix(const float transform_array[16]) {
    base_T_mem_ = bev_utils::createTransformMatrix(transform_array);
    ROS_INFO("base_T_mem updated!");
}

void BEVPublisher::setTransformMatrix(const Eigen::Matrix4f& transform_matrix) {
    base_T_mem_ = transform_matrix;
    ROS_INFO("base_T_mem updated!");
}

void BEVPublisher::setBEVConfig(const bev_utils::BEVConfig& config) {
    bev_config_ = config;
    ROS_INFO("BEV config updated: %dx%d grid", config.grid_width, config.grid_height);
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
    
    ROS_INFO("LaserScan params updated: frame_id=%s, angle=[%.2f, %.2f], range=[%.2f, %.2f]",
             frame_id_.c_str(), angle_min_, angle_max_, range_min_, range_max_);
}

void BEVPublisher::publishPersonStates(const std::vector<STrack>& stracks, ros::Time stamp) {
    if (stracks.empty()) {
        return;
    }
    
    emma_safe_msgs::PersonStateArray state_array;
    state_array.header.stamp = stamp;
    state_array.header.frame_id = "odom";
    state_array.header.seq = published_count_;
    
    for (const auto& strack : stracks) {
        emma_safe_msgs::PersonState person_state;
        
        // 获取位置和速度信息
        std::vector<float> tlwh = strack.tlwh;                
        // odom坐标系        
        person_state.id = strack.track_id;
        person_state.x = tlwh[0] + tlwh[2] / 2;
        person_state.y = tlwh[1] + tlwh[3] / 2; 
        person_state.vx = strack.mean[4];
        person_state.vy = strack.mean[5];
        person_state.height = 0.30;  // 默认高度
        person_state.width = 0.30;  // 使用较大的维度作为宽度
        
        // 协方差矩阵
        person_state.covariances.clear();
        person_state.covariances.reserve(strack.covariance.size());
        for (int i = 0; i < strack.covariance.size(); ++i) {
            person_state.covariances.push_back(strack.covariance.data()[i]);
        }

        state_array.people_state.push_back(person_state);
    }
    
    stracks_pub_.publish(state_array);
    ROS_DEBUG("发布了 %zu 个人物状态", stracks.size());
}

void BEVPublisher::publishBEVResult(const rknpu2::float16* bev_result, ros::Time stamp) {  
    try {
        // TODO: CenterPoint 点从mem转换至ref再转换至base再转换至odom 
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_.lookupTransform(
                "odom",   // 目标坐标系
                "base_footprint",  // 源坐标系
                stamp,             // 获取最c新可用变换
                ros::Duration(0)        // 超时时间
            );
        } catch (tf2::TransformException &ex) {
            transform = tf_buffer_.lookupTransform(
                "odom",   // 目标坐标系
                "base_footprint",  // 源坐标系
                ros::Time(0),             // 获取最新可用变换
                ros::Duration(0)        // 超时时间
            );
        }
        Eigen::Matrix4f odom_T_base = tf2::transformToEigen(transform).matrix().cast<float>();
        Eigen::Matrix4f odom_T_mem = odom_T_base * base_T_mem_;
        Eigen::Matrix4f mem_T_odom = odom_T_mem.inverse();

        std::vector<Object> objects = bev_utils::getBEVBboxOdom(bev_result, odom_T_mem, 96, 96); // 可视化

        // std::vector<CenterPoint> center_points_odom = bev_utils::transformCenterPoint(center_points, odom_T_mem);
        // std::vector<CenterPoint>转换至std::vector<Object>
        // std::vector<Object> objects = bev_utils::centerPointsToObjects(center_points_odom);

        std::vector<STrack> output_stracks = tracker.update(objects);

        // 发布人物状态
        publishPersonStates(output_stracks, stamp);

        cv::Mat bev_img = bev_utils::get_bev_image(bev_result, 96, 96);
        cv::Mat bev_img_resized;
        cv::resize(bev_img, bev_img_resized, cv::Size(960, 960), 0, 0, cv::INTER_NEAREST);
        float scale_factor = 960.0f / 96.0f;  // 10倍缩放

        for (int i = 0; i < output_stracks.size(); i++)
        {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            float vx = output_stracks[i].mean[4];
            float vy = output_stracks[i].mean[5];
            ROS_INFO("object %d vel: vx=%.3f, vy=%.3f", output_stracks[i].track_id, vx, vy);

            Eigen::Vector4f homogeneous_point_odom(tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, 0, 1.0f);
            // 应用变换矩阵
            Eigen::Vector4f transformed = mem_T_odom * homogeneous_point_odom;
            Scalar s = tracker.get_color(output_stracks[i].track_id);

            // 缩放坐标以适应 960x960 图像
            float scaled_x = transformed[0] * scale_factor;
            float scaled_y = transformed[2] * scale_factor;  // 注意：这里应该是 transformed[1] 而不是 transformed[2]
            
            // 调整文本和矩形的大小
            cv::putText(bev_img_resized, 
                        cv::format("id=%d vx=%.1f vy=%.1f", output_stracks[i].track_id, vx * 100., vy * 100.), 
                        cv::Point(scaled_x, scaled_y - 8),  // 调整位置
                        0, 0.8,  // 增大字体大小
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

            cv::rectangle(bev_img_resized, cv::Rect(scaled_x - tlwh[2] / 2, scaled_y - tlwh[3] / 2, tlwh[2], tlwh[3]), s, 5);
        }            
        writer.write(bev_img_resized);
        // 转换BEV结果为LaserScan
        sensor_msgs::LaserScan scan = bev_utils::pointCloudToLaserScan(
            bev_utils::transformPointCloud(
                bev_utils::bevGridToPointCloud(bev_result, bev_config_),
                base_T_mem_
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
        ROS_DEBUG("BEV LaserScan published #%d", published_count_);
        
        
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