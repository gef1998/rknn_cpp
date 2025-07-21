#ifndef ROS2OPENCV_H
#define ROS2OPENCV_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "rknnPool.hpp"
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <geometry_msgs/Pose.h>

template <typename rknnModel, typename inputType, typename outputType>
class Ros2OpenCV {
public:
    Ros2OpenCV(rknnPool<rknnModel, inputType, outputType>& pool,
            const std::string& topic_name, int threadNum);
    ~Ros2OpenCV();
    
private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    int frames = 0;
    int threadNum = 6;
    long long beforeTime = 0;
    long long afterTime = 0;
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;
    ros::Publisher bbox_pub_;  // 新增发布器
    rknnPool<rknnModel, inputType, outputType>& pool_;
    std::string topic_name_;
    outputType outputData;
};

template<typename rknnModel, typename inputType, typename outputType>
Ros2OpenCV<rknnModel, inputType, outputType>::Ros2OpenCV(rknnPool<rknnModel, inputType, outputType>& pool,
    const std::string& topic_name, int threadNum)
    : it_(nh_), pool_(pool), topic_name_(topic_name), threadNum(threadNum) {
    sub_ = it_.subscribe(topic_name_, 10, &Ros2OpenCV::imageCallback, this);
    bbox_pub_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>("Bbox", 10);  // 新增发布器初始化
}

template <typename rknnModel, typename inputType, typename outputType>
Ros2OpenCV<rknnModel, inputType, outputType>::~Ros2OpenCV() {
    cv::destroyWindow("Camera FPS"); 
}

template<typename rknnModel, typename inputType, typename outputType>
void Ros2OpenCV<rknnModel, inputType, outputType>::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    cv::Mat frame = cv_ptr->image;
    if (pool_.put(frame) != 0) {
        ROS_WARN("Failed to submit frame to RKNN pool");
        return;
    }
    if (frames == 0) { 
        struct timeval time;
        gettimeofday(&time, nullptr);       
        beforeTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        cv::namedWindow("Camera FPS", cv::WINDOW_NORMAL);
        cv::resizeWindow("Camera FPS", 40, 40);
    }

    while (ros::ok()) {
        outputData.image = frame;
        if (pool_.get(outputData) != 0)
                return;
        frames++;
        cv::imshow("Camera FPS", outputData.image);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
            return;
        jsk_recognition_msgs::BoundingBoxArray bbox_array;
        bbox_array.header = msg->header;
        bbox_array.header.stamp = stamp;
        bbox_array.header.frame_id = "bbox_array";
        for (int i = 0; i < outputData.detection.count; i++)
        {
            detect_result_t *det_result = &(outputData.detection.results[i]);
            jsk_recognition_msgs::BoundingBox bbox;
            bbox.header = msg->header;
            bbox.header.stamp = stamp;
            bbox.header.frame_id = "bbox";
            bbox.pose.position.x = (det_result->box.left + det_result->box.right) / 2.0;
            bbox.pose.position.y = (det_result->box.top + det_result->box.bottom) / 2.0;
            bbox.pose.position.z = 0;

            // 设置尺寸（单位：米）
            bbox.dimensions.x = det_result->box.right - det_result->box.left; // 宽度
            bbox.dimensions.y = det_result->box.bottom - det_result->box.top; // 高度
            bbox.dimensions.z = 0;  // 深度

            // 设置类别和置信度
            bbox.label = outputData.detection.id;     // 类别ID需与训练模型一致

            bbox_array.boxes.push_back(bbox);
        }
        bbox_pub_.publish(bbox_array);  // 发布边界框消息
        
        if (frames % 120 == 0) {
            struct timeval time;
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120 frames average frame rate:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }
}

// template<typename rknnModel, typename inputType, typename outputType>
// void Ros2OpenCV<rknnModel, inputType, outputType>::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
//     ros::Time::waitForValid();
//     auto stamp = ros::Time::now(); 
//     std::cout << "now stamp: " << stamp << ", msg stamp: " << msg->header.stamp
//     << ", diff:" << (stamp.toSec() - msg->header.stamp.toSec()) << "s." << std::endl;

//     cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
//     cv::Mat frame = cv_ptr->image;
//     if (pool_.put(frame) != 0) {
//         ROS_WARN("Failed to submit frame to RKNN pool");
//         return;
//     }
//     if (frames == 0) { 
//         struct timeval time;
//         gettimeofday(&time, nullptr);       
//         beforeTime = time.tv_sec * 1000 + time.tv_usec / 1000;
//         cv::namedWindow("Camera FPS", cv::WINDOW_NORMAL);
//         cv::resizeWindow("Camera FPS", 240, 240);
//     }
//     static auto last_fps_time = std::chrono::steady_clock::now();
//     static int frame_counter = 0;
//     frame_counter++;

//     outputData.image = frame;
//     if (pool_.tryGet(outputData, 200)) { // 50ms超时
//         cv::imshow("Camera FPS", outputData.image);
//         if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
//             return;
//         jsk_recognition_msgs::BoundingBoxArray bbox_array;
//         bbox_array.header.stamp = ros::Time(stamp.sec, stamp.nsec);
//         bbox_array.header.frame_id = "bbox_array";
//         // ROS_INFO("stamp: %d.%d", bbox_array.header.stamp.sec, bbox_array.header.stamp.nsec);
//         for (int i = 0; i < outputData.detection.count; i++)
//         {
//             detect_result_t *det_result = &(outputData.detection.results[i]);
//             jsk_recognition_msgs::BoundingBox bbox;
//             bbox.header.stamp = ros::Time(stamp.sec, stamp.nsec);
//             bbox.header.frame_id = "bbox";
//             bbox.pose.position.x = (det_result->box.left + det_result->box.right) / 2.0;
//             bbox.pose.position.y = (det_result->box.top + det_result->box.bottom) / 2.0;
//             bbox.pose.position.z = 0;

//             // 设置尺寸（单位：米）
//             bbox.dimensions.x = det_result->box.right - det_result->box.left; // 宽度
//             bbox.dimensions.y = det_result->box.bottom - det_result->box.top; // 高度
//             bbox.dimensions.z = 0;  // 深度

//             // 设置类别和置信度
//             bbox.label = outputData.detection.id;     // 类别ID需与训练模型一致

//             bbox_array.boxes.push_back(bbox);
//         }
//         bbox_pub_.publish(bbox_array);  // 发布边界框消息

//         const auto now = std::chrono::steady_clock::now();
//         const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();

//             if (elapsed > 1000) { // 每秒更新一次
//                 const double fps = frame_counter * 1000.0 / elapsed;
//                 ROS_INFO_STREAM("Current FPS: " << std::fixed << std::setprecision(2) << fps);
                
//                 // 更新窗口标题显示FPS
//                 cv::setWindowTitle("Camera FPS", "Camera FPS - " + std::to_string(fps) + " FPS");
                
//                 frame_counter = 0;
//                 last_fps_time = now;
//         }
//     }
// }


#endif // ROS2OPENCV_H
