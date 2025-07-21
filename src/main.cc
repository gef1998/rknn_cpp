#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include "ros2opencv.h"
#include <image_transport/image_transport.h>
#include <chrono>

typedef struct {
    char* model_path;
    char* video_input;
    int thread_num;
    bool help_flag;
} ProgramArgs;

void print_usage(const char* prog_name) {
    printf("Usage: %s <rknn_model> <video_input> [options]\n", prog_name);
    printf("Options:\n");
    printf("  -t <num>    Set thread number (default: 6)\n");
    printf("  -h          Show this help message\n");
}

ProgramArgs parse_arguments(int argc, char** argv) {
    ProgramArgs args = {0};
    args.thread_num = 6;
    args.help_flag = false;

    if (argc < 3) {
        args.help_flag = true;
        return args;
    }

    args.model_path = argv[1];
    args.video_input = argv[2];

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            args.thread_num = atoi(argv[++i]);
            if (args.thread_num <= 0) {
                fprintf(stderr, "Warning: Invalid thread number, using default 6\n");
                args.thread_num = 6;
            }
        } else if (strcmp(argv[i], "-h") == 0) {
            args.help_flag = true;
        }
    }
    return args;
}    


int main(int argc, char **argv)
{
    ProgramArgs args = parse_arguments(argc, argv);
    ros::init(argc, argv, "object_detection");

    if (args.help_flag) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    printf("Model: %s\nVideo: %s\nThreads: %d\n",
           args.model_path, args.video_input, args.thread_num);
    
    const char* model_name = args.model_path;
    const char* video_name = args.video_input;
    int threadNum = args.thread_num;

    rknnPool<rkYolov5s, cv::Mat, ImgBbox> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }
    Ros2OpenCV<rkYolov5s, cv::Mat, ImgBbox> processor(testPool, "/pallet/color/image_raw", threadNum); 
    ros::spin();
    return 0;
}