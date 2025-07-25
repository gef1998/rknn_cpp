# SimpleBEV 多摄像头ROS集成

本项目实现了SimpleBEV模型与8个摄像头的ROS集成，将多目图像合并为统一输入进行推理。

## 功能特性

- **多摄像头同步订阅**: 同时订阅8个摄像头的image_raw topics
- **自动图像预处理**: 将输入图像调整为224x400分辨率
- **图像合并**: 将8个摄像头图像合并为8×224×400×3格式
- **RKNN推理集成**: 无缝集成SimpleBEV RKNN推理流程
- **实时性能监控**: 提供FPS统计和性能监控

## 订阅的Topics

节点会订阅以下8个摄像头topics：

```
/back/left/image_raw       # 后左摄像头
/back/right/image_raw      # 后右摄像头  
/front/left/image_raw      # 前左摄像头
/front/right/image_raw     # 前右摄像头
/left/left/image_raw       # 左左摄像头
/left/right/image_raw      # 左右摄像头
/right/left/image_raw      # 右左摄像头
/right/right/image_raw     # 右右摄像头
```

## 数据格式

- **输入**: 8个独立的image_raw消息 (sensor_msgs/Image)
- **预处理**: 每个图像调整为224×400×3 (BGR格式)
- **合并输出**: 8×224×400×3 = 2,150,400字节的uint8数组

## 编译

```bash
cd rknn_simplebev
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 使用方法

### 1. 基本启动

```bash
# 直接启动
./rknn_simplebev_ros encoder.rknn grid_sample.rknn flat_idx.bin decoder.rknn

# 使用启动脚本
chmod +x launch_multicamera.sh
./launch_multicamera.sh model/encoder.rknn model/grid_sample.rknn model/flat_idx.bin model/decoder.rknn
```

### 2. 设置ROS参数

```bash
# 设置RKNN线程池大小
rosparam set /rknn_simplebev_multicamera/thread_num 2

# 启动节点
./rknn_simplebev_ros encoder.rknn grid_sample.rknn flat_idx.bin decoder.rknn
```

### 3. 使用launch文件 (可选)

创建launch文件 `multicamera_simplebev.launch`:

```xml
<launch>
  <!-- SimpleBEV多摄像头节点 -->
  <node name="rknn_simplebev_multicamera" pkg="rknn_cpp" type="rknn_simplebev_ros" output="screen">
    <param name="thread_num" value="1"/>
    <arg name="encoder_model" value="$(find rknn_cpp)/rknn_simplebev/model/encoder.rknn"/>
    <arg name="grid_sample_model" value="$(find rknn_cpp)/rknn_simplebev/model/grid_sample.rknn"/>
    <arg name="flat_idx_file" value="$(find rknn_cpp)/rknn_simplebev/model/flat_idx.bin"/>
    <arg name="decoder_model" value="$(find rknn_cpp)/rknn_simplebev/model/decoder.rknn"/>
  </node>
</launch>
```

## 性能监控

节点运行时会显示以下信息：

- **多摄像头处理FPS**: 每100帧统计一次
- **推理FPS**: 每120帧统计一次  
- **总体平均FPS**: 程序结束时显示

## 故障排除

### 1. Topics未发布

```bash
# 检查可用topics
rostopic list | grep image_raw

# 检查topic频率
rostopic hz /back/left/image_raw
```

### 2. 图像同步问题

如果8个摄像头时间戳不完全同步，可以调整同步策略：

```cpp
// 在MultiCameraSubscriber构造函数中调整
sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(20),  // 增加队列大小
    // ... 其他参数
);
```

### 3. 内存使用过高

减少线程数量或调整缓冲区大小：

```bash
rosparam set /rknn_simplebev_multicamera/thread_num 1
```

### 4. 推理性能问题

- 检查RKNN模型是否正确加载
- 验证输入数据格式是否正确
- 监控系统资源使用情况

## 代码结构

```
rknn_simplebev/
├── include/
│   ├── multi_camera_subscriber.hpp  # 多摄像头订阅器
│   └── simplebev.hpp                # SimpleBEV推理引擎
├── src/
│   ├── multi_camera_subscriber.cpp  # 多摄像头实现
│   ├── main.cc                  # ROS节点主程序
│   └── simplebev.cc                 # SimpleBEV实现
├── launch_multicamera.sh            # 启动脚本
└── README_MULTICAMERA.md            # 本文档
```

## 注意事项

1. **摄像头同步**: 确保8个摄像头的时间戳尽可能同步
2. **网络带宽**: 8个图像流需要足够的网络带宽  
3. **计算资源**: 图像预处理和推理需要充足的CPU/GPU资源
4. **内存管理**: 合理设置线程池大小避免内存溢出

## 技术支持

如遇问题，请检查：

1. ROS环境是否正确配置
2. 所有依赖包是否已安装 (cv_bridge, image_transport, message_filters)
3. RKNN模型文件是否存在且格式正确
4. 摄像头topics是否正常发布

---

更多信息请参考项目主README文件。 