#ifndef MULTI_SENSOR_DATA_HPP
#define MULTI_SENSOR_DATA_HPP
#include "fp16/Float16.h"

// 多传感器数据结构，包含图像和点云数据
struct MultiSensorData {
    unsigned char* image_data;
    rknpu2::float16* pointcloud_data;
    
    MultiSensorData(unsigned char* img_data, rknpu2::float16* pc_data) 
        : image_data(img_data), pointcloud_data(pc_data) {}
        
    MultiSensorData() : image_data(nullptr), pointcloud_data(nullptr) {}
};

#endif // MULTI_SENSOR_DATA_HPP 