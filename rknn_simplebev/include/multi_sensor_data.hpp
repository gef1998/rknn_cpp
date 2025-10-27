#ifndef MULTI_SENSOR_DATA_HPP
#define MULTI_SENSOR_DATA_HPP
#include "fp16/Float16.h"
#include <ros/time.h>  // 添加 ROS 时间头文件

// 多传感器数据结构，包含图像和点云数据
struct MultiSensorData {
    unsigned char* image_data;
    rknpu2::float16* pointcloud_data;
    ros::Time stamp;
    
    MultiSensorData(unsigned char* img_data, rknpu2::float16* pc_data, ros::Time stamp) 
        : image_data(img_data), pointcloud_data(pc_data), stamp(stamp) {}
        
    MultiSensorData() : image_data(nullptr), pointcloud_data(nullptr), stamp(ros::Time(0)) {}
};

#endif // MULTI_SENSOR_DATA_HPP ccc