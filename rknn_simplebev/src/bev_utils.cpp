#include "bev_utils.hpp"
#include <ros/ros.h>
#include <cmath>
#include <algorithm>

namespace bev_utils {
// simple bev中bev_grid为[[0,0,0],[1,0,0] ...]，所以x为j，y为0，z为i 见self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
std::vector<Point3D> bevGridToPointCloud(const rknpu2::float16* bev_grid, 
                                         const BEVConfig& config,
                                         float threshold) {
    std::vector<Point3D> points;
    points.reserve(config.grid_width * config.grid_height / 4); // 预估障碍物点数量
    
    for (int i = 0; i < config.grid_height; ++i) {
        for (int j = 0; j < config.grid_width; ++j) {
            int idx = i * config.grid_width + j;
            float value = static_cast<float>(bev_grid[idx]);
            
            // 如果值大于阈值，认为是障碍物
            if (value > threshold) {
                // 将网格坐标转换为物理坐标
                // 网格[0,0]对应物理坐标的左上角
                float x = j; 
                float y = 0.0f; // BEV是2D，y设为0
                float z = i;
                
                points.emplace_back(x, y, z);
            }
        }
    }
    
    ROS_INFO("BEV网格转换为点云: 从 %dx%d 网格中提取了 %zu 个障碍物点", 
             config.grid_width, config.grid_height, points.size());
    
    return points;
}

std::vector<Point3D> transformPointCloud(const std::vector<Point3D>& points,
                                         const Eigen::Matrix4f& transform_matrix) {
    std::vector<Point3D> transformed_points;
    transformed_points.reserve(points.size());
    
    for (const auto& point : points) {
        // 将3D点转换为齐次坐标
        Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1.0f);
        
        // 应用变换矩阵
        Eigen::Vector4f transformed = transform_matrix * homogeneous_point;
        
        // 转换回3D坐标
        transformed_points.emplace_back(
            transformed[0],
            transformed[1], 
            transformed[2]
        );
    }
    
    ROS_INFO("坐标变换完成: 变换了 %zu 个点", points.size());
    
    return transformed_points;
}

sensor_msgs::LaserScan pointCloudToLaserScan(const std::vector<Point3D>& points,
                                            const std::string& frame_id,
                                            float angle_min,
                                            float angle_max,
                                            float angle_increment,
                                            float range_min,
                                            float range_max) {
    sensor_msgs::LaserScan scan;
    
    // 设置LaserScan消息头
    scan.header.stamp = ros::Time::now();
    scan.header.frame_id = frame_id;
    
    // 设置扫描参数
    scan.angle_min = angle_min;
    scan.angle_max = angle_max;
    scan.angle_increment = angle_increment;
    scan.range_min = range_min;
    scan.range_max = range_max;
    
    // 计算角度数量
    int num_angles = static_cast<int>((angle_max - angle_min) / angle_increment) + 1;
    scan.ranges.resize(num_angles, range_max); // 初始化为最大距离
    scan.intensities.resize(num_angles, 0.0);
    
    // 将点云转换为激光扫描数据
    for (const auto& point : points) {
        // 计算距离和角度
        float range = std::sqrt(point.x * point.x + point.y * point.y);
        float angle = std::atan2(point.y, point.x);
        
        // 检查距离是否在有效范围内
        if (range < range_min || range > range_max) {
            continue;
        }
        
        // 检查角度是否在扫描范围内
        if (angle < angle_min || angle > angle_max) {
            continue;
        }
        
        // 计算角度索引
        int angle_idx = static_cast<int>((angle - angle_min) / angle_increment);
        if (angle_idx >= 0 && angle_idx < num_angles) {
            // 如果这个角度上已经有更近的点，保留更近的
            if (range < scan.ranges[angle_idx]) {
                scan.ranges[angle_idx] = range;
                scan.intensities[angle_idx] = 1.0; // 可以根据需要设置强度值
            }
        }
    }
    
    ROS_INFO("点云转换为LaserScan: %zu 个点转换为 %d 个激光束", 
             points.size(), num_angles);
    
    return scan;
}

Eigen::Matrix4f createTransformMatrix(const float matrix_array[16]) {
    Eigen::Matrix4f transform;
    
    // 按行填充矩阵 (行优先)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transform(i, j) = matrix_array[i * 4 + j];
        }
    }
    
    return transform;
}

Eigen::Matrix4f createTransformMatrix(const Eigen::Matrix3f& rotation, 
                                     const Eigen::Vector3f& translation) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    // 设置旋转部分
    transform.block<3, 3>(0, 0) = rotation;
    
    // 设置平移部分
    transform.block<3, 1>(0, 3) = translation;
    
    return transform;
}

} // namespace bev_utils 