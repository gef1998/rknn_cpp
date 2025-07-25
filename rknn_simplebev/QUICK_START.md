# SimpleBEV多摄像头功能快速开始

## 快速测试步骤

### 1. 编译项目
```bash
cd rknn_simplebev
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
```

### 2. 准备模型文件
确保你有以下模型文件：
- `encoder.rknn` - 编码器模型
- `grid_sample.rknn` - 网格采样模型  
- `flat_idx.bin` - 扁平索引文件
- `decoder.rknn` - 解码器模型

### 3. 启动测试摄像头发布器
在第一个终端中运行：
```bash
# 启动ROS master
roscore
```

在第二个终端中运行：
```bash
# 启动测试摄像头发布器
cd rknn_simplebev
python3 test_camera_publisher.py
```

### 4. 启动SimpleBEV节点
在第三个终端中运行：
```bash
cd rknn_simplebev/build
../launch_multicamera.sh ../model/encoder.rknn ../model/grid_sample.rknn ../model/flat_idx.bin ../model/decoder.rknn
```

## 验证运行

### 检查topics
```bash
# 查看图像topics
rostopic list | grep image_raw

# 检查图像发布频率
rostopic hz /back/left/image_raw

# 查看图像信息
rostopic info /back/left/image_raw
```

### 监控性能
SimpleBEV节点会输出以下信息：
- 多摄像头处理FPS
- 推理FPS
- 总体平均FPS

### 检查日志
```bash
# 查看ROS日志
roslog list
```

## 故障排除

### 常见问题

1. **编译错误**
   - 检查ROS环境：`echo $ROS_DISTRO`
   - 安装依赖：`sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-image-transport`

2. **模型文件不存在**
   - 确认模型文件路径正确
   - 检查文件权限

3. **图像topics未发布**
   - 确认测试发布器正在运行
   - 检查网络连接

4. **推理失败**
   - 检查RKNN运行时库
   - 验证模型格式

### 调试命令
```bash
# 查看节点状态
rosnode list
rosnode info /rknn_simplebev_multicamera

# 检查参数
rosparam list | grep simplebev
```

## 下一步

测试成功后，你可以：
1. 替换测试发布器为真实摄像头数据
2. 调整图像预处理参数
3. 优化推理性能
4. 集成到你的应用中

更多详细信息请参考 `README_MULTICAMERA.md`。 