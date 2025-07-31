#ifndef BEV_PUBLISHER_HPP
#define BEV_PUBLISHER_HPP

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <Eigen/Dense>
#include "bev_utils.hpp"

/**
 * BEV结果发布器类
 * 负责将SimpleBEV的推理结果转换为LaserScan消息并发布
 */
class BEVPublisher {
public:
    /**
     * 构造函数
     * @param nh: ROS节点句柄
     * @param topic_name: 发布话题名称，默认为"/bev_perception/grid_pc"
     * @param queue_size: 消息队列大小
     */
    explicit BEVPublisher(ros::NodeHandle& nh, 
                         const std::string& topic_name = "/bev_perception/grid_pc",
                         int queue_size = 10);

    /**
     * 设置base_T_ref变换矩阵
     * @param transform_array: 4x4变换矩阵（行优先）
     */
    void setTransformMatrix(const float transform_array[16]);

    /**
     * 设置base_T_ref变换矩阵
     * @param transform_matrix: Eigen 4x4变换矩阵
     */
    void setTransformMatrix(const Eigen::Matrix4f& transform_matrix);

    /**
     * 设置BEV配置参数
     * @param config: BEV配置结构
     */
    void setBEVConfig(const bev_utils::BEVConfig& config);

    /**
     * 设置LaserScan参数
     * @param frame_id: 坐标系名称
     * @param angle_min: 最小角度
     * @param angle_max: 最大角度
     * @param angle_increment: 角度增量
     * @param range_min: 最小距离
     * @param range_max: 最大距离
     */
    void setLaserScanParams(const std::string& frame_id = "base_link",
                           float angle_min = -M_PI,
                           float angle_max = M_PI,
                           float angle_increment = M_PI/180.0,
                           float range_min = 0.1f,
                           float range_max = 30.0f);

    /**
     * 发布BEV结果为LaserScan消息
     * @param simplebev: SimpleBEV实例引用
     * @param bev_result: BEV推理结果
     */
    void publishBEVResult(const rknpu2::float16* bev_result);

    /**
     * 处理传感器数据并发布结果（完整流程）
     * @param simplebev: SimpleBEV实例引用
     * @param sensor_data: 传感器数据
     */

    /**
     * 获取当前发布器统计信息
     */
    void printStats() const;

private:
    ros::NodeHandle& nh_;
    ros::Publisher laser_pub_;
    
    // 变换矩阵和配置
    Eigen::Matrix4f base_T_ref_;
    bev_utils::BEVConfig bev_config_;
    
    // LaserScan参数
    std::string frame_id_;
    float angle_min_, angle_max_, angle_increment_;
    float range_min_, range_max_;
    
    // 统计信息
    mutable int published_count_;
    mutable ros::Time last_publish_time_;
    
    std::string topic_name_;
};

#endif // BEV_PUBLISHER_HPP 