#include <ros/ros.h>
#include <signal.h>
#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "simplebev.hpp"
#include "rknnPool.hpp"
#include "multi_sensor_subscriber.hpp"
#include "multi_sensor_data.hpp"
#include "fp16/Float16.h"
#include "bev_publisher.hpp"

// 全局变量
std::unique_ptr<rknnPool<SimpleBEV, MultiSensorData, rknpu2::float16*>> g_pool;
std::unique_ptr<MultiSensorSubscriber> g_sensor_subscriber;
bool g_running = true;

// 信号处理函数
void signalHandler(int signal) {
    ROS_INFO("Received signal %d, shutting down...", signal);
    g_running = false;
    ros::shutdown();
}

// 多传感器数据处理回调函数
void sensorProcessingCallback(unsigned char* image_data, rknpu2::float16* pointcloud_data) {
    if (!g_pool || !g_running) {
        return;
    }
    
    // 创建多传感器数据结构
    MultiSensorData sensor_data(image_data, pointcloud_data);
    
    // 提交数据到RKNN线程池进行推理
    if (g_pool->put(sensor_data) != 0) {
        ROS_WARN("Fail to put sensor data to rknn pool!");
        return;
    }
}

void visualize_bev_grid(rknpu2::float16* bev_data, int width, int height) {
    // 创建OpenCV矩阵
    cv::Mat bev_image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // 对每个像素进行sigmoid处理和二值化
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;            
            float raw_val = static_cast<float>(bev_data[idx]);                        
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

int main(int argc, char **argv) {
    // 初始化ROS节点
    ros::init(argc, argv, "rknn_simplebev_multi_sensor");
    ros::NodeHandle nh;
    BEVPublisher bev_publisher(nh, "/bev_perception/grid_pc");
    // 2. 设置您的base_T_ref变换矩阵
    const float base_T_ref[16] = {9.5396e-04f,  -1.2006e-03f, 9.9983e-02f, -4.7392e+00f,
                                -9.9907e-02f, -3.1694e-03f, 8.8558e-04f,  4.6638e+00f,
                                4.2110e-03f,  -7.4923e-02f, -1.6396e-03f, 2.6543e-01f,
                                0.0f,  0.0f,  0.0f,  1.0f};
    bev_publisher.setTransformMatrix(base_T_ref);

    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    ROS_INFO("Starting SimpleBEV Multi-Sensor ROS Node (8 Cameras + 2 LiDARs)");
    
    // 检查命令行参数
    if (argc < 5) {
        ROS_ERROR("用法: %s <encoder_model> <grid_sample_model> <flat_idx_file> <decoder_model>", argv[0]);
        return -1;
    }
    
    // 配置模型路径
    SimpleBEV::ModelPaths modelPaths{
        argv[1], // encoder
        argv[2], // grid_sample
        argv[4], // decoder
        argv[3],  // flat_idx
        argv[5]  // lasernet_path
    };
    
    ROS_INFO("Model path configuration:");
    ROS_INFO("  Encoder: %s", modelPaths.encoder_path.c_str());
    ROS_INFO("  Grid Sample: %s", modelPaths.grid_sample_path.c_str());
    ROS_INFO("  Decoder: %s", modelPaths.decoder_path.c_str());
    ROS_INFO("  Flat Index: %s", modelPaths.flat_idx_path.c_str());
    ROS_INFO("  LaserNet: %s", modelPaths.lasernet_path.c_str());
    
    // 从ROS参数服务器获取线程数
    int threadNum = 3;
    ROS_INFO("RKNN thread pool size: %d", threadNum);
    
    try {
        // 初始化RKNN线程池
        ROS_INFO("Initializing RKNN thread pool for multi-sensor data...");
        g_pool = std::make_unique<rknnPool<SimpleBEV, MultiSensorData, rknpu2::float16*>>(
            modelPaths, threadNum);
        
        if (g_pool->init() != 0) {
            ROS_ERROR("RKNN thread pool initialization failed!");
            return -1;
        }
        ROS_INFO("RKNN thread pool initialized successfully");
        
        // 创建多传感器订阅器
        ROS_INFO("Creating multi-sensor subscriber...");
        g_sensor_subscriber = std::make_unique<MultiSensorSubscriber>(
            nh, sensorProcessingCallback
        );
        
        // 启动传感器订阅
        g_sensor_subscriber->start();
        ROS_INFO("Multi-sensor subscription started successfully");
        ROS_INFO("Expected input format:");
        ROS_INFO("  Images: 8 x 224 x 400 x 3 = %d bytes", 
                 MultiSensorSubscriber::TOTAL_IMAGE_SIZE);
        ROS_INFO("  Point cloud: %d points x 3 dimensions = %d floats", 
                 MultiSensorSubscriber::POINTCLOUD_SIZE, 
                 MultiSensorSubscriber::TOTAL_POINTCLOUD_SIZE);
        
        // 统计信息
        struct timeval time;
        gettimeofday(&time, nullptr);
        auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        int processed_frames = 0;
        auto beforeTime = startTime;
        
        ROS_INFO("SimpleBEV multi-sensor node ready, waiting for sensor data...");
        
        ros::Rate loop_rate(5);
        while (ros::ok() && g_running) {
            ros::spinOnce();
            
            // 处理队列中的推理结果
            rknpu2::float16 * result;
            if (g_pool->get(result, 100) == 0) {
                processed_frames++;

                // 显示BEV可视化结果
                if (result != nullptr) {
                    visualize_bev_grid(result, 96, 96);
                    bev_publisher.publishBEVResult(result);
                }
                
                // 每120帧打印一次统计信息
                if (processed_frames % 120 == 0) {
                    gettimeofday(&time, nullptr);
                    auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
                    double fps = 120.0 / (double)(currentTime - beforeTime) * 1000.0;
                    ROS_INFO("Multi-sensor inference FPS: %.2f fps", fps);
                    beforeTime = currentTime;
                }
            }
            loop_rate.sleep();
        }
        
        // 计算总体统计信息
        gettimeofday(&time, nullptr);
        auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        if (processed_frames > 0) {
            double total_fps = (double)processed_frames / (double)(endTime - startTime) * 1000.0;
            ROS_INFO("Average multi-sensor inference FPS: %.2f fps", total_fps);
            ROS_INFO("Total processed frames: %d", processed_frames);
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Program runtime exception: %s", e.what());
        return -1;
    }
    
    // 清理资源
    ROS_INFO("Cleaning up resources...");
    
    if (g_sensor_subscriber) {
        g_sensor_subscriber->stop();
        g_sensor_subscriber.reset();
    }
    
    if (g_pool) {
        // 清空线程池中剩余的任务
        int remaining_results = 0;
        rknpu2::float16 * result;
        while (g_pool->get(result) == 0) {
            remaining_results++;
        }
        if (result != nullptr) {
            visualize_bev_grid(result, 96, 96);
        }
        if (remaining_results > 0) {
            ROS_INFO("Processed remaining %d inference results", remaining_results);
        }
        g_pool.reset();
    }
    
    ROS_INFO("SimpleBEV multi-sensor node exited");
    return 0;
} 