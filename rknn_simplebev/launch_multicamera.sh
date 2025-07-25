#!/bin/bash

# SimpleBEV多摄像头ROS节点启动脚本
# 用法: ./launch_multicamera.sh [encoder_model] [grid_sample_model] [flat_idx_file] [decoder_model]

set -e

echo "=========================================="
echo "  SimpleBEV 多摄像头ROS节点启动脚本"
echo "=========================================="

# 检查参数数量
if [ $# -ne 4 ]; then
    echo "错误: 需要4个参数"
    echo "用法: $0 <encoder_model> <grid_sample_model> <flat_idx_file> <decoder_model>"
    echo ""
    echo "示例:"
    echo "  $0 model/encoder.rknn model/grid_sample.rknn model/flat_idx.bin model/decoder.rknn"
    exit 1
fi

# 获取参数
ENCODER_MODEL="$1"
GRID_SAMPLE_MODEL="$2"
FLAT_IDX_FILE="$3"
DECODER_MODEL="$4"

# 检查文件是否存在
echo "检查模型文件..."
for file in "$ENCODER_MODEL" "$GRID_SAMPLE_MODEL" "$FLAT_IDX_FILE" "$DECODER_MODEL"; do
    if [ ! -f "$file" ]; then
        echo "错误: 文件不存在: $file"
        exit 1
    fi
    echo "  ✓ $file"
done

echo ""
echo "配置信息:"
echo "  Encoder 模型: $ENCODER_MODEL"
echo "  Grid Sample 模型: $GRID_SAMPLE_MODEL"  
echo "  Flat Index 文件: $FLAT_IDX_FILE"
echo "  Decoder 模型: $DECODER_MODEL"
echo ""

# 检查ROS环境
if [ -z "$ROS_MASTER_URI" ]; then
    echo "警告: ROS环境未设置，尝试source setup.bash..."
    if [ -f "/opt/ros/melodic/setup.bash" ]; then
        source /opt/ros/melodic/setup.bash
    elif [ -f "/opt/ros/noetic/setup.bash" ]; then
        source /opt/ros/noetic/setup.bash
    else
        echo "错误: 无法找到ROS环境"
        exit 1
    fi
fi

echo "ROS环境:"
echo "  ROS_MASTER_URI: $ROS_MASTER_URI"
echo "  ROS_DISTRO: ${ROS_DISTRO:-未知}"
echo ""

# 启动节点
echo "启动SimpleBEV多摄像头ROS节点..."
echo "订阅的topics:"
echo "  /back/left/image_raw"
echo "  /back/right/image_raw"
echo "  /front/left/image_raw"
echo "  /front/right/image_raw"
echo "  /left/left/image_raw"
echo "  /left/right/image_raw"
echo "  /right/left/image_raw"
echo "  /right/right/image_raw"
echo "按 Ctrl+C 停止节点"
echo "=========================================="
# 启动节点
./install/rknn_simplebev_Linux/rknn_simplebev "$ENCODER_MODEL" "$GRID_SAMPLE_MODEL" "$FLAT_IDX_FILE" "$DECODER_MODEL"
