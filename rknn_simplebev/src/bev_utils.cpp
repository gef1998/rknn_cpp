#include "bev_utils.hpp"
#include <ros/ros.h>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace bev_utils {
// simple bev中bev_grid为[[0,0,0],[1,0,0] ...]，所以x为j，y为0，z为i 见self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
std::vector<Point3D> bevGridToPointCloud(const rknpu2::float16* bev_grid, 
                                         const BEVConfig& config,
                                         float threshold) {
    std::vector<Point3D> points;
    const int W = config.grid_width;
    const int H = config.grid_height;

    points.reserve(W * H / 4); // 预估障碍物点数量
    const rknpu2::float16* ptr = bev_grid;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j, ++ptr) {
            float value = static_cast<float>(*ptr);
            
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
    
    ROS_INFO("BEV grid to point cloud: Extracted %zu obstacle points from %dx%d grid", 
            points.size(), W, H);
    
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
    
    // ROS_INFO("坐标变换完成: 变换了 %zu 个点", points.size());
    
    return transformed_points;
}



std::vector<CenterPoint> transformCenterPoint(const std::vector<CenterPoint>& points,
                                         const Eigen::Matrix4f& transform_matrix) {
    std::vector<CenterPoint> transformed_points;
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
            transformed[2],
            point.conf
        );
    }
    
    // ROS_INFO("坐标变换完成: 变换了 %zu 个点", points.size());
    
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
    
    // ROS_INFO("点云转换为LaserScan: %zu 个点转换为 %d 个激光束", 
    //          points.size(), num_angles);
    
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


void visualize_bev_grid(const rknpu2::float16* bev_data, int width, int height) {
    // 创建OpenCV矩阵
    cv::Mat bev_image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    const rknpu2::float16* ptr = bev_data;
    // 对每个像素进行sigmoid处理和二值化
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++, ptr++) {
            float raw_val = static_cast<float>(*ptr);                        
            // 二值化：>0.5为不可行驶区域(白色)，<=0.5为可行驶区域(黑色)
            // uchar pixel_val = (raw_val > 0.0f) ? 255 : 0;            
            // 设置RGB值
            if (raw_val > 0.0f) {
                bev_image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255); // 不可行驶区域
            }
        }
    }
    // 在中心添加红色标记点，便于观察方向
    int center_x = width / 2;
    int center_y = height / 2;
    cv::rectangle(bev_image, 
        cv::Point(center_x-2, center_y-2),
        cv::Point(center_x+2, center_y+2),
        cv::Scalar(0, 0, 255),
        -1);
    // 直接显示图像
    cv::imshow("BEV Inference with Multi-Sensor", bev_image);
    cv::waitKey(1); // 非阻塞显示，允许实时更新
    // printf("BEV网格已显示 (白色=不可行驶，黑色=可行驶) - 集成激光数据\n");
}

cv::Mat get_bev_image(const rknpu2::float16* bev_data, int width, int height) {
    // 创建OpenCV矩阵
    cv::Mat bev_image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    const rknpu2::float16* ptr = bev_data;
    // 对每个像素进行sigmoid处理和二值化
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++, ptr++) {
            float raw_val = static_cast<float>(*ptr);                        
            // 二值化：>0.5为不可行驶区域(白色)，<=0.5为可行驶区域(黑色)
            // uchar pixel_val = (raw_val > 0.0f) ? 255 : 0;            
            // 设置RGB值
            if (raw_val > 0.0f) {
                bev_image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255); // 不可行驶区域
            }
        }
    }
    // 在中心添加红色标记点，便于观察方向
    int center_x = width / 2;
    int center_y = height / 2;
    cv::rectangle(bev_image, 
        cv::Point(center_x-2, center_y-2),
        cv::Point(center_x+2, center_y+2),
        cv::Scalar(0, 0, 255),
        -1);
    return bev_image;
}

std::vector<CenterPoint> peakLocalMax(const cv::Mat& img, int min_distance, int threshold_abs) {
    CV_Assert(img.type() == CV_8UC1);

    // 1. 找局部峰值
    cv::Mat dilated;
    cv::dilate(img, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE,
                 cv::Size(2 * min_distance + 1, 2 * min_distance + 1)));
    cv::Mat local_max = (img == dilated) & (img >= threshold_abs);

    std::vector<cv::Point> points;
    cv::findNonZero(local_max, points);

    // 2. 将峰值按强度排序
    std::vector<std::pair<cv::Point, int>> sorted_points;
    sorted_points.reserve(points.size());
    for (const auto& p : points)
        sorted_points.push_back({p, img.at<uchar>(p)});

    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });

    // 3. NMS
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
    std::vector<CenterPoint> centers;
    for (const auto& sp : sorted_points) {
        const cv::Point& p = sp.first;
        if (mask.at<uchar>(p) == 0) {
            centers.push_back({static_cast<float>(p.x),
                               0.0f,
                               static_cast<float>(p.y),
                               static_cast<float>(sp.second)/255.f});
            // 标记周围 min_distance 范围为排除
            cv::circle(mask, p, min_distance * 2, cv::Scalar(1), -1);
        }
    }

    return centers;
}


std::vector<CenterPoint> getCenterPoint(const rknpu2::float16* bev_grid, 
                                    int W, int H) {
    cv::Mat center_image(H, W, CV_8UC1, cv::Scalar(0));
    const rknpu2::float16* ptr = bev_grid + W * H; // TODO: 这里写死了center分支所属channel，需要和网络输出对齐

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j, ptr++) {
            float value = static_cast<float>(*ptr);
            center_image.at<uchar>(i, j) =  1.0 / (1.0 + expf(-value)) * 255.;
        }
    }
    cv::morphologyEx(center_image, center_image, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
    std::vector<CenterPoint> center_points = peakLocalMax(center_image, 2, 128);
    return center_points;
}

std::vector<Object> getBEVBboxOdom(const rknpu2::float16* bev_grid, 
                                    const Eigen::Matrix4f& odom_T_mem,
                                    int W, int H) {
    cv::Mat center_image(H, W, CV_8UC1, cv::Scalar(0));
    const rknpu2::float16* ptr = bev_grid + W * H; // TODO: 这里写死了center分支所属channel，需要和网络输出对齐

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j, ptr++) {
            float value = static_cast<float>(*ptr);
            center_image.at<uchar>(i, j) =  1.0 / (1.0 + expf(-value)) * 255.;
        }
    }
    cv::morphologyEx(center_image, center_image, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
    
    // 使用轮廓检测来获取边界框
    cv::Mat binary_image;
    cv::threshold(center_image, binary_image, 127, 255, cv::THRESH_BINARY);
    binary_image.convertTo(binary_image, CV_8U);
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<Object> objects;
    objects.reserve(contours.size() * 2);

    for (const auto& cnt : contours) {
        cv::Rect rect = cv::boundingRect(cnt);
        int x = rect.x;
        int y = rect.y;
        int w = rect.width;
        int h = rect.height;
        
        // 过滤太小的轮廓 (面积 <= 9)
        if (w * h <= 9) {
            continue;
        }
        // 如果面积 <= 16，设置固定大小
        if (w * h <= 16) {
            Object obj;
            Eigen::Vector4f homogeneous_point(x + w / 2.0f, 0.0f, y + h / 2.0f, 1.0f);
            // 应用变换矩阵
            Eigen::Vector4f transformed = odom_T_mem * homogeneous_point;
            obj.rect.x = transformed[0] - 3;  // 左上角 x
            obj.rect.y = transformed[1] - 3;  // 左上角 y
            obj.rect.width = 6;
            obj.rect.height = 6;
            obj.label = 0;                 // 默认标签
            obj.prob = 1;            // 概率赋值, 目前默认为1
            objects.push_back(obj);
            continue;
        }
        
        if (w * h >= 64) {
            // 对于大轮廓，使用局部最大值搜索
            cv::Rect roi_rect(x, y, w, h);
            if (roi_rect.x >= 0 && roi_rect.y >= 0 && 
                roi_rect.x + roi_rect.width <= center_image.cols && 
                roi_rect.y + roi_rect.height <= center_image.rows) {
                
                cv::Mat roi = center_image(roi_rect);
                std::vector<CenterPoint> local_peaks = peakLocalMax(roi, 2, 180);
                
                for (const auto& peak : local_peaks) {
                    // Python 代码中：coordinates = coordinates + np.array([y, x]) 把坐标加上 ROI 偏移
                    // 然后 x = cx - 3, y = cy - 3，绘制时左上角是 (x-1, y-1)
                    // 所以最终框的左上角是 (cx-4, cy-4)，右下角是 (cx+2, cy+2)，中心是 (cx-1, cy-1)
                    int cx = static_cast<int>(peak.x) + x;  // 加上 ROI 的 x 偏移
                    int cy = static_cast<int>(peak.z) + y;  // 加上 ROI 的 y 偏移，CenterPoint 的 z 对应 y 坐标
                    Eigen::Vector4f homogeneous_point(cx, 0.0f, cy, 1.0f);
                    // 应用变换矩阵
                    Eigen::Vector4f transformed = odom_T_mem * homogeneous_point;
        
                    Object obj;
                    obj.rect.x = transformed[0] - 3;  // 左上角 x
                    obj.rect.y = transformed[1] - 3;  // 左上角 y
                    obj.rect.width = 6;
                    obj.rect.height = 6;
                    obj.label = 0;                 // 默认标签
                    obj.prob = 1;            // 概率赋值, 目前默认为1
                    objects.push_back(obj);
                }
            }
        } else {
            // 对于中等轮廓 (面积在 17-63 之间)，直接使用外接矩形
            Object obj;
            Eigen::Vector4f homogeneous_point(rect.x + rect.width / 2.0f, 0.0f, rect.y + rect.height / 2.0f, 1.0f);
            // 应用变换矩阵
            Eigen::Vector4f transformed = odom_T_mem * homogeneous_point;
            obj.rect.x = transformed[0] - rect.width / 2.0f;  // 左上角 x
            obj.rect.y = transformed[1] - rect.height / 2.0f;  // 左上角 y
            obj.rect.width = rect.width;
            obj.rect.height = rect.height;
            obj.label = 0;                 // 默认标签
            obj.prob = 1;            // 概率赋值, 目前默认为1
            objects.push_back(obj);
        }
    }
    return objects;
}

std::vector<Object> centerPointsToObjects(const std::vector<CenterPoint>& centers) {
    std::vector<Object> objects;
    objects.reserve(centers.size());

    const float w = 30.0f * 0.05;
    const float h = 30.0f * 0.05;

    for (const auto& cp : centers) {
        Object obj;
        obj.rect.x = cp.x - w / 2.0f;  // 左上角 x
        obj.rect.y = cp.y - h / 2.0f;  // 左上角 y
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = 0;                 // 默认标签
        obj.prob = cp.conf;            // 概率赋值
        objects.push_back(obj);
    }

    return objects;
}

} // namespace bev_utils 