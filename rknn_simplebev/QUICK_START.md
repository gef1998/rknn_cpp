# SimpleBEV 快速开始指南

## 🚀 快速集成BEV到LaserScan发布功能

### 1. 最简单的使用方式

```cpp
#include <ros/ros.h>
#include "bev_publisher.hpp"
#include "simplebev.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "my_bev_node");
    ros::NodeHandle nh;
    
    // ✅ 第1步：创建BEV发布器
    BEVPublisher bev_publisher(nh, "/bev_perception/grid_pc");
    
    // ✅ 第2步：设置您的变换矩阵
    const float base_T_ref[16] = {
        9.5396e-04f,  -1.2006e-03f, 9.9983e-02f, -4.7392e+00f,
        -9.9907e-02f, -3.1694e-03f, 8.8558e-04f,  4.6638e+00f,
        4.2110e-03f,  -7.4923e-02f, -1.6396e-03f, 2.6543e-01f,
        0.0f,  0.0f,  0.0f,  1.0f
    };
    bev_publisher.setTransformMatrix(base_T_ref);
    
    // ✅ 第3步：在您现有的推理循环中添加发布
    ros::Rate rate(10);
    while (ros::ok()) {
        // 您现有的SimpleBEV推理代码
        rknpu2::float16* bev_result = your_simplebev.infer_multi_sensor(image_data, pointcloud_data);
        
        // 🎯 只需添加这一行：发布结果
        if (bev_result != nullptr) {
            bev_publisher.publishBEVResult(your_simplebev, bev_result);
        }
        
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
}
```

### 2. 查看发布结果

```bash
# 启动您的节点
rosrun your_package your_bev_node

# 查看发布的LaserScan数据
rostopic echo /bev_perception/grid_pc

# 检查发布频率
rostopic hz /bev_perception/grid_pc
```

### 3. 在RViz中可视化

1. 启动RViz：`rviz`
2. 添加LaserScan显示
3. 设置Topic为：`/bev_perception/grid_pc`
4. 设置Fixed Frame为：`base_link`

## 📁 需要包含的头文件

```cpp
#include "bev_publisher.hpp"  // BEV发布器
#include "simplebev.hpp"      // 您现有的SimpleBEV类
```

## 🔧 CMakeLists.txt 配置

```cmake
# 添加新的源文件
add_executable(your_bev_node
    src/your_main.cpp
    rknn_simplebev/src/bev_publisher.cpp
    rknn_simplebev/src/bev_utils.cpp
    # ... 其他源文件
)

# 包含头文件路径
target_include_directories(your_bev_node PRIVATE
    rknn_simplebev/include
    # ... 其他包含路径
)

# 链接库
target_link_libraries(your_bev_node
    ${catkin_LIBRARIES}
    # ... 其他库
)
```

## ⚡ 关键点

1. **无需修改现有推理代码** - 只需在推理后添加发布调用
2. **自动处理坐标转换** - 使用您提供的base_T_ref矩阵
3. **标准ROS话题** - 发布到 `/bev_perception/grid_pc`
4. **96×96网格支持** - 自动处理您的BEV网格格式

## 🎯 您只需要做的事情

1. ✅ 包含头文件：`#include "bev_publisher.hpp"`
2. ✅ 创建发布器：`BEVPublisher bev_publisher(nh, "/bev_perception/grid_pc");`
3. ✅ 设置变换矩阵：`bev_publisher.setTransformMatrix(base_T_ref);`
4. ✅ 发布结果：`bev_publisher.publishBEVResult(simplebev, bev_result);`

就这么简单！🎉 