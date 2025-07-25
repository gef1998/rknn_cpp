#include <ros/ros.h>
#include <signal.h>
#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include "simplebev.hpp"
#include "rknnPool.hpp"
#include "multi_camera_subscriber.hpp"

// 全局变量
std::unique_ptr<rknnPool<SimpleBEV, unsigned char*, int>> g_pool;
std::unique_ptr<MultiCameraSubscriber> g_camera_subscriber;
bool g_running = true;

// 信号处理函数
void signalHandler(int signal) {
    ROS_INFO("Received signal %d, shutting down...", signal);
    g_running = false;
    ros::shutdown();
}

// 图像处理回调函数
void imageProcessingCallback(unsigned char* merged_image_data) {
    if (!g_pool || !g_running) {
        return;
    }
    
    // 提交图像数据到RKNN线程池进行推理
    if (g_pool->put(merged_image_data) != 0) {
        ROS_WARN("Fail to put image to rknn pool!");
        return;
    }
    
    // 尝试获取推理结果
    int result;
    if (g_pool->get(result) == 0) {
        ROS_DEBUG("Inference Competed, res: %d", result);
    }
}

int main(int argc, char **argv) {
    // 初始化ROS节点
    ros::init(argc, argv, "rknn_simplebev_multicamera");
    ros::NodeHandle nh;
    
    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    ROS_INFO("Starting SimpleBEV Multi-Camera ROS Node");
    
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
        argv[3]  // flat_idx
    };
    
    ROS_INFO("Model path configuration:");
    ROS_INFO("  Encoder: %s", modelPaths.encoder_path.c_str());
    ROS_INFO("  Grid Sample: %s", modelPaths.grid_sample_path.c_str());
    ROS_INFO("  Decoder: %s", modelPaths.decoder_path.c_str());
    ROS_INFO("  Flat Index: %s", modelPaths.flat_idx_path.c_str());
    
    // 从ROS参数服务器获取线程数
    int threadNum = 1;
    ROS_INFO("RKNN thread pool size: %d", threadNum);
    
    try {
        // 初始化RKNN线程池
        ROS_INFO("Initializing RKNN thread pool...");
        g_pool = std::make_unique<rknnPool<SimpleBEV, unsigned char*, int>>(
            modelPaths, threadNum);
        
        if (g_pool->init() != 0) {
            ROS_ERROR("RKNN thread pool initialization failed!");
            return -1;
        }
        ROS_INFO("RKNN thread pool initialized successfully");
        
        // 创建多摄像头订阅器
        ROS_INFO("Creating multi-camera subscriber...");
        g_camera_subscriber = std::make_unique<MultiCameraSubscriber>(
            nh, imageProcessingCallback
        );
        
        // 启动摄像头订阅
        g_camera_subscriber->start();
        ROS_INFO("Multi-camera subscription started successfully");
        
        // 统计信息
        struct timeval time;
        gettimeofday(&time, nullptr);
        auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        int processed_frames = 0;
        auto beforeTime = startTime;
        
        ROS_INFO("SimpleBEV multi-camera node ready, waiting for image data...");
        ROS_INFO("Expected input format: 8 x 224 x 400 x 3 = %d bytes", 
                 MultiCameraSubscriber::TOTAL_SIZE);
        
        // 主循环
        ros::Rate loop_rate(5); // 100Hz TODO:对帧率的影响
        while (ros::ok() && g_running) {
            ros::spinOnce();
            
            // 处理队列中的推理结果
            int result;
            while (g_pool->get(result) == 0) {
                processed_frames++;
                
                // 每120帧打印一次统计信息
                if (processed_frames % 120 == 0) {
                    gettimeofday(&time, nullptr);
                    auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
                    double fps = 120.0 / (double)(currentTime - beforeTime) * 1000.0;
                    ROS_INFO("Inference FPS: %.2f fps", fps);
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
            ROS_INFO("Average inference FPS: %.2f fps", total_fps);
            ROS_INFO("Total processed frames: %d", processed_frames);
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Program runtime exception: %s", e.what());
        return -1;
    }
    
    // 清理资源
    ROS_INFO("Cleaning up resources...");
    
    if (g_camera_subscriber) {
        g_camera_subscriber->stop();
        g_camera_subscriber.reset();
    }
    
    if (g_pool) {
        // 清空线程池中剩余的任务
        int remaining_results = 0;
        int result;
        while (g_pool->get(result) == 0) {
            remaining_results++;
        }
        if (remaining_results > 0) {
            ROS_INFO("Processed remaining %d inference results", remaining_results);
        }
        g_pool.reset();
    }
    
    ROS_INFO("SimpleBEV multi-camera node exited");
    return 0;
} 