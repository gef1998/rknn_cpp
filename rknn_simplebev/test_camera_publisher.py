#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleBEV多摄像头测试发布器
发布8个虚拟摄像头数据用于测试多摄像头订阅功能
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time

class TestCameraPublisher:
    def __init__(self):
        rospy.init_node('test_camera_publisher', anonymous=True)
        
        self.bridge = CvBridge()
        
        # 8个摄像头topic名称
        self.camera_topics = [
            '/back/left/image_raw',
            '/back/right/image_raw', 
            '/front/left/image_raw',
            '/front/right/image_raw',
            '/left/left/image_raw',
            '/left/right/image_raw',
            '/right/left/image_raw',
            '/right/right/image_raw'
        ]
        
        # 创建发布器
        self.publishers = {}
        for topic in self.camera_topics:
            self.publishers[topic] = rospy.Publisher(topic, Image, queue_size=10)
        
        # 图像尺寸
        self.width = 400
        self.height = 224
        
        # 发布频率
        self.rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("测试摄像头发布器初始化完成")
        rospy.loginfo("发布topics:")
        for topic in self.camera_topics:
            rospy.loginfo(f"  {topic}")
            
    def generate_test_image(self, camera_idx, frame_count):
        """生成测试图像"""
        # 创建彩色测试图像
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 不同摄像头使用不同颜色
        colors = [
            (255, 0, 0),    # 蓝色 - back/left
            (0, 255, 0),    # 绿色 - back/right
            (0, 0, 255),    # 红色 - front/left
            (255, 255, 0),  # 青色 - front/right
            (255, 0, 255),  # 品红 - left/left
            (0, 255, 255),  # 黄色 - left/right
            (128, 128, 128), # 灰色 - right/left
            (255, 128, 0)   # 橙色 - right/right
        ]
        
        color = colors[camera_idx % len(colors)]
        image[:, :] = color
        
        # 添加文本标识
        camera_names = [
            "BACK-L", "BACK-R", "FRONT-L", "FRONT-R",
            "LEFT-L", "LEFT-R", "RIGHT-L", "RIGHT-R"
        ]
        
        # 摄像头名称
        cv2.putText(image, camera_names[camera_idx], 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # 帧计数
        cv2.putText(image, f"Frame: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # 时间戳
        cv2.putText(image, f"Time: {time.time():.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1)
        
        # 添加移动的矩形作为动态元素
        rect_x = int((frame_count * 2) % (self.width - 50))
        cv2.rectangle(image, (rect_x, 120), (rect_x + 50, 170), (0, 0, 0), 2)
        
        return image
        
    def publish_images(self):
        """发布图像"""
        frame_count = 0
        
        rospy.loginfo("开始发布测试图像...")
        rospy.loginfo(f"图像尺寸: {self.width}x{self.height}")
        rospy.loginfo("按 Ctrl+C 停止发布")
        
        while not rospy.is_shutdown():
            timestamp = rospy.Time.now()
            
            # 为每个摄像头生成并发布图像
            for i, topic in enumerate(self.camera_topics):
                # 生成测试图像
                test_image = self.generate_test_image(i, frame_count)
                
                # 转换为ROS图像消息
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(test_image, "bgr8")
                    img_msg.header.stamp = timestamp
                    img_msg.header.frame_id = f"camera_{i}"
                    
                    # 发布图像
                    self.publishers[topic].publish(img_msg)
                    
                except Exception as e:
                    rospy.logerr(f"发布图像失败 {topic}: {e}")
            
            frame_count += 1
            
            # 每100帧打印一次状态
            if frame_count % 100 == 0:
                rospy.loginfo(f"已发布 {frame_count} 帧图像")
            
            self.rate.sleep()
            
    def run(self):
        """运行发布器"""
        try:
            self.publish_images()
        except rospy.ROSInterruptException:
            rospy.loginfo("测试摄像头发布器已停止")
        except Exception as e:
            rospy.logerr(f"发布器运行错误: {e}")

if __name__ == '__main__':
    try:
        publisher = TestCameraPublisher()
        publisher.run()
    except Exception as e:
        print(f"启动失败: {e}") 