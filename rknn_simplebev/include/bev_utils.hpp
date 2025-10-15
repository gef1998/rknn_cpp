#ifndef BEV_UTILS_HPP
#define BEV_UTILS_HPP

#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <Eigen/Dense>
#include "fp16/Float16.h"

namespace bev_utils {

// BEV网格配置参数
struct BEVConfig {
    int grid_width = 96;        // 网格宽度
    int grid_height = 96;       // 网格高度
};

// 3D点结构
struct Point3D {
    float x, y, z;
    Point3D(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

/**
 * 将BEV网格转换为3D点云
 * @param bev_grid: 96x96的BEV网格数据，值>0表示障碍物，<0表示可通行
 * @param config: BEV配置参数
 * @param threshold: 障碍物阈值，>threshold的网格被认为是障碍物
 * @return 障碍物点的3D坐标向量
 */
std::vector<Point3D> bevGridToPointCloud(const rknpu2::float16* bev_grid, 
                                         const BEVConfig& config = BEVConfig(),
                                         float threshold = 0.0f);

/**
 * 应用4x4变换矩阵到点云
 * @param points: 输入点云
 * @param transform_matrix: 4x4变换矩阵 (base_T_ref)
 * @return 变换后的点云
 */
std::vector<Point3D> transformPointCloud(const std::vector<Point3D>& points,
                                         const Eigen::Matrix4f& transform_matrix);

/**
 * 将3D点云转换为LaserScan消息
 * @param points: 3D点云
 * @param laser_config: LaserScan配置参数
 * @param frame_id: 坐标系名称
 * @return LaserScan消息
 */
sensor_msgs::LaserScan pointCloudToLaserScan(const std::vector<Point3D>& points,
                                            const std::string& frame_id = "base_footprint",
                                            float angle_min = -M_PI,
                                            float angle_max = M_PI,
                                            float angle_increment = M_PI/180.0,  // 1度
                                            float range_min = 0.1f,
                                            float range_max = 30.0f);

/**
 * 从4x4矩阵数组创建Eigen变换矩阵
 * @param matrix_array: 4x4矩阵的一维数组表示(行优先)
 * @return Eigen 4x4矩阵
 */
Eigen::Matrix4f createTransformMatrix(const float matrix_array[16]);

/**
 * 从旋转和平移创建变换矩阵
 * @param rotation: 3x3旋转矩阵
 * @param translation: 3x1平移向量
 * @return 4x4变换矩阵
 */
Eigen::Matrix4f createTransformMatrix(const Eigen::Matrix3f& rotation, 
                                     const Eigen::Vector3f& translation);

} // namespace bev_utils

#endif // BEV_UTILS_HPP 