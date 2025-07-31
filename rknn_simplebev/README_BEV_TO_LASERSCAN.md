# SimpleBEV BEV网格到LaserScan转换功能

本功能可以将SimpleBEV的96×96 BEV网格输出转换为ROS LaserScan消息，支持坐标变换。

## 功能特性

- **BEV网格处理**: 将96×96的BEV网格转换为3D点云
- **坐标变换**: 支持4×4变换矩阵（base_T_ref）进行坐标系转换  
- **LaserScan生成**: 将点云数据转换为ROS标准的LaserScan消息
- **外部发布**: BEV推理结果在外部处理，支持发布到指定ROS话题
- **自动发布器**: 提供BEVPublisher类，自动处理转换和发布流程
- **灵活配置**: 支持自定义BEV网格参数和LaserScan参数
- **话题发布**: 默认发布到 `/bev_perception/grid_pc` 话题

## 文件结构

```
rknn_simplebev/
├── include/
│   ├── bev_utils.hpp          # BEV处理工具函数声明
│   ├── bev_publisher.hpp      # BEV发布器类声明
│   └── simplebev.hpp          # 更新的SimpleBEV类定义
├── src/
│   ├── bev_utils.cpp          # BEV处理工具函数实现
│   ├── bev_publisher.cpp      # BEV发布器类实现
│   └── simplebev.cc           # 更新的SimpleBEV类实现
├── examples/
│   ├── bev_publisher_example.cpp     # 完整的BEV发布示例
│   ├── simple_bev_usage.cpp          # 简化的使用示例
│   └── bev_to_laserscan_example.cpp  # 原始使用示例
└── README_BEV_TO_LASERSCAN.md # 本说明文档
```

## 使用方法

### 1. 使用BEVPublisher（推荐方式）

```cpp
#include <ros/ros.h>
#include "simplebev.hpp"
#include "bev_publisher.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "bev_node");
    ros::NodeHandle nh;
    
    // 1. 创建BEV发布器，自动发布到 /bev_perception/grid_pc
    BEVPublisher bev_publisher(nh, "/bev_perception/grid_pc");
    
    // 2. 设置您的base_T_ref变换矩阵
    const float base_T_ref[16] = {
        9.5396e-04f,  -1.2006e-03f, 9.9983e-02f, -4.7392e+00f,
        -9.9907e-02f, -3.1694e-03f, 8.8558e-04f,  4.6638e+00f,
        4.2110e-03f,  -7.4923e-02f, -1.6396e-03f, 2.6543e-01f,
        0.0f,  0.0f,  0.0f,  1.0f
    };
    bev_publisher.setTransformMatrix(base_T_ref);
    
    // 3. 初始化SimpleBEV
    SimpleBEV::ModelPaths paths;
    // ... 设置模型路径 ...
    SimpleBEV simplebev(paths);
    // ... 初始化SimpleBEV ...
    
    // 4. 主循环
    ros::Rate rate(10);
    while (ros::ok()) {
        // 执行推理
        rknpu2::float16* bev_result = simplebev.infer_multi_sensor(image_data, pointcloud_data);
        
        // 发布结果（自动转换为LaserScan并发布到话题）
        if (bev_result != nullptr) {
            bev_publisher.publishBEVResult(simplebev, bev_result);
        }
        
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
}
```

### 2. 直接使用SimpleBEV方法

```cpp
#include "simplebev.hpp"

// 1. 创建SimpleBEV实例并初始化
SimpleBEV::ModelPaths paths;
paths.encoder_path = "path/to/encoder.rknn";
paths.grid_sample_path = "path/to/grid_sample.rknn";
paths.decoder_path = "path/to/decoder.rknn";
paths.lasernet_path = "path/to/lasernet.rknn";
paths.flat_idx_path = "path/to/flat_idx.bin";

SimpleBEV simplebev(paths);
// ... 初始化SimpleBEV ...

// 2. 执行推理得到BEV结果
MultiSensorData sensor_data(image_data, pointcloud_data);
rknpu2::float16* bev_result = simplebev.infer(sensor_data);

// 3. 设置base_T_ref变换矩阵
const float base_T_ref[16] = { /* 您的变换矩阵 */ };

// 4. 转换为LaserScan
sensor_msgs::LaserScan scan = simplebev.bevToLaserScan(
    bev_result, 
    base_T_ref,
    bev_utils::BEVConfig(),  // 使用默认配置
    "base_link"              // 目标坐标系
);

// 5. 发布LaserScan消息
laser_pub.publish(scan);
```

### 2. 使用数组形式的变换矩阵

```cpp
// 如果您的变换矩阵是数组形式
float transform_array[16] = {
    1.0f, 0.0f, 0.0f, 0.1f,  // 第一行
    0.0f, 1.0f, 0.0f, 0.2f,  // 第二行
    0.0f, 0.0f, 1.0f, 0.0f,  // 第三行
    0.0f, 0.0f, 0.0f, 1.0f   // 第四行
};

sensor_msgs::LaserScan scan = simplebev.bevToLaserScan(
    bev_result, 
    transform_array,
    bev_utils::BEVConfig(),
    "base_link"
);
```

### 3. 自定义BEV配置

```cpp
// 自定义BEV网格配置
bev_utils::BEVConfig config;
config.grid_width = 96;
config.grid_height = 96;
config.physical_width = 48.0f;   // 48米物理宽度
config.physical_height = 48.0f;  // 48米物理高度
config.resolution = 0.5f;        // 0.5米/网格
config.center_x = 24.0f;         // 网格中心x坐标
config.center_y = 24.0f;         // 网格中心y坐标

sensor_msgs::LaserScan scan = simplebev.bevToLaserScan(
    bev_result, 
    base_T_ref, 
    config,      // 使用自定义配置
    "base_link"
);
```

## 配置参数说明

### BEVConfig参数

- `grid_width/grid_height`: BEV网格尺寸（默认96×96）
- `physical_width/physical_height`: 网格覆盖的物理区域尺寸（米）
- `resolution`: 网格分辨率（米/网格）
- `center_x/center_y`: 网格中心对应的物理坐标（米）

### LaserScan参数（在pointCloudToLaserScan函数中配置）

- `angle_min/angle_max`: 扫描角度范围（弧度）
- `angle_increment`: 角度分辨率（弧度）
- `range_min/range_max`: 距离测量范围（米）

## 坐标系说明

### BEV网格坐标系
- 网格[0,0]对应物理坐标的左上角
- x轴向右为正，y轴向下为正
- 网格中心通常对应车辆位置

### 变换矩阵 base_T_ref
- 4×4齐次变换矩阵
- 将点从ref坐标系变换到base坐标系
- 矩阵格式：
  ```
  [R11 R12 R13 Tx]
  [R21 R22 R23 Ty]
  [R31 R32 R33 Tz]
  [ 0   0   0   1]
  ```

### LaserScan坐标系
- 以指定的frame_id为参考
- 角度从-π到π，0度对应x轴正方向
- 距离为从原点到障碍物的欧几里得距离

## 编译依赖

确保您的项目包含以下依赖：

```cmake
# CMakeLists.txt
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
)

find_package(Eigen3 REQUIRED)

# 包含头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  rknn_simplebev/include
)

# 链接库
target_link_libraries(your_target
  ${catkin_LIBRARIES}
  # 其他库...
)
```

## 注意事项

1. **坐标系一致性**: 确保base_T_ref变换矩阵正确反映了您的坐标系关系
2. **网格参数**: BEV网格的物理尺寸和分辨率需要与您的模型训练参数一致
3. **障碍物阈值**: 默认阈值为0.0，可以根据需要调整
4. **性能考虑**: 对于实时应用，可以考虑减少点云密度或优化角度分辨率

## 示例运行

编译并运行示例：

```bash
# 编译
catkin build

# 运行完整示例
rosrun your_package bev_publisher_example

# 运行简化示例
rosrun your_package simple_bev_usage

# 查看发布的LaserScan消息
rostopic echo /bev_perception/grid_pc

# 查看话题信息
rostopic info /bev_perception/grid_pc

# 实时监控发布频率
rostopic hz /bev_perception/grid_pc
```

## 故障排除

1. **编译错误**: 检查Eigen3和ROS依赖是否正确安装
2. **空的LaserScan**: 检查BEV网格是否包含障碍物数据，或调整阈值
3. **坐标错误**: 验证变换矩阵和BEV配置参数是否正确
4. **性能问题**: 考虑降低LaserScan的角度分辨率或距离范围

如有问题，请检查ROS日志获取详细错误信息。

## 更新说明

### v2.0 更新（外部发布版本）

- **重构设计**: 将bevToLaserScan调用从infer方法内部移到外部
- **新增BEVPublisher类**: 专门负责BEV结果的转换和发布
- **话题发布**: 默认发布到 `/bev_perception/grid_pc` 话题
- **简化坐标处理**: 直接使用网格坐标，移除物理尺寸参数
- **支持您的变换矩阵**: 集成您提供的具体base_T_ref变换矩阵

### 主要变化

1. **infer方法**: 现在只返回BEV推理结果，不再内部处理转换
2. **外部发布**: 通过BEVPublisher类或直接调用bevToLaserScan方法
3. **坐标系**: 简化为直接使用网格坐标 (x=j, y=0, z=i)
4. **话题标准化**: 统一发布到 `/bev_perception/grid_pc`

这样的设计让您可以：
- 更灵活地控制何时发布结果
- 在发布前对BEV结果进行额外处理
- 支持多个不同的发布器或后处理流程
- 更好地集成到现有的ROS系统中 