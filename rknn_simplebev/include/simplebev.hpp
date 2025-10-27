#ifndef SIMPLEBEV_H
#define SIMPLEBEV_H

#include "rknn_api.h"
#include <mutex>
#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include "fp16/Float16.h"
#include "multi_sensor_data.hpp"
#include "bev_utils.hpp"
#include <sensor_msgs/LaserScan.h>
#include <Eigen/Dense>

// 常量定义
constexpr int kEncoderInputNum = 1;
constexpr int kLaserNetInputNum = 1;
constexpr int kGridSampleInputNum = 2;
constexpr int kDecoderInputNum = 2;


constexpr int kEncoderOutputNum = 1;
constexpr int kLaserNetOutputNum = 1;
constexpr int kGridSampleOutputNum = 2;
constexpr int kDecoderOutputNum = 1;

// 类型别名
using InputAttrArrEncoder = std::array<rknn_tensor_attr, kEncoderInputNum>;
using InputAttrArrGridSample = std::array<rknn_tensor_attr, kGridSampleInputNum>;
using InputAttrArrDecoder = std::array<rknn_tensor_attr, kDecoderInputNum>;
using InputAttrArrLaserNet = std::array<rknn_tensor_attr, kLaserNetInputNum>;

using OutputAttrArrEncoder = std::array<rknn_tensor_attr, kEncoderOutputNum>;
using OutputAttrArrGridSample = std::array<rknn_tensor_attr, kGridSampleOutputNum>;
using OutputAttrArrDecoder = std::array<rknn_tensor_attr, kDecoderOutputNum>;
using OutputAttrArrLaserNet = std::array<rknn_tensor_attr, kLaserNetOutputNum>;

using TensorMemArrEncoder = std::array<rknn_tensor_mem*, kEncoderOutputNum>;
using TensorMemArrLaserNet = std::array<rknn_tensor_mem*, kLaserNetOutputNum>;
using TensorMemArrGridSample = std::array<rknn_tensor_mem*, kGridSampleOutputNum>;
using TensorMemArrDecoder = std::array<rknn_tensor_mem*, kDecoderOutputNum>;

// 工具函数声明
static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

struct InferResult {
    rknpu2::float16 *output;
    ros::Time stamp;
    InferResult() : output(nullptr), stamp(ros::Time(0)) {}
    InferResult(rknpu2::float16 *output, ros::Time stamp) : output(output), stamp(stamp) {}
};

// SimpleBEV类定义
class SimpleBEV {
public:
    struct ModelPaths {
        std::string encoder_path, grid_sample_path, decoder_path, flat_idx_path, lasernet_path;
    };

    explicit SimpleBEV(const ModelPaths& paths);

    int init(rknn_context *encoder_ctx_in,
             rknn_context *grid_sample_ctx_in,
             rknn_context *decoder_ctx_in,
             rknn_context *lasernet_ctx_in,
             rknn_tensor_mem* flat_idx_mems_in,
             bool share_weight);

    int init_flat_idx_mems();
    rknn_context* get_encoder_pctx();
    rknn_context* get_grid_sample_pctx();
    rknn_context* get_decoder_pctx();
    rknn_context* get_lasernet_pctx();

    rknn_tensor_mem* get_flat_idx_mems();
    int get_input_size() const;

    int query_model_io_num(rknn_context ctx, rknn_input_output_num &io_num, const char* ctx_name);
    int query_input_attributes(rknn_context ctx, rknn_input_output_num& io_num, const char* model_name, rknn_tensor_attr* attrs);
    int query_output_attributes(rknn_context ctx, rknn_input_output_num& io_num, const char* model_name, rknn_tensor_attr* attrs);

    // rknpu2::float16* infer(unsigned char* input_data);

    // 修改：infer方法现在接收图像和点云数据
    InferResult infer_multi_sensor(unsigned char* image_data, rknpu2::float16* pointcloud_data, ros::Time stamp);
    
    // 重载：使用MultiSensorData结构的infer方法
    InferResult infer(const MultiSensorData& sensor_data);
    
    // BEV网格处理和LaserScan转换方法
    /**
     * 将BEV推理结果转换为LaserScan消息
     * @param bev_result: BEV推理结果(96x96网格)
     * @param base_T_mem: 坐标变换矩阵(4x4)
     * @param config: BEV配置参数
     * @param frame_id: 目标坐标系
     * @return LaserScan消息
     */
    sensor_msgs::LaserScan bevToLaserScan(const rknpu2::float16* bev_result,
                                         const Eigen::Matrix4f& base_T_mem,
                                         const bev_utils::BEVConfig& config = bev_utils::BEVConfig(),
                                         const std::string& frame_id = "base_footprint");
    
    /**
     * 便捷方法：从矩阵数组创建变换矩阵并转换为LaserScan
     * @param bev_result: BEV推理结果
     * @param transform_array: 4x4变换矩阵(行优先数组)
     * @param config: BEV配置参数
     * @param frame_id: 目标坐标系
     * @return LaserScan消息
     */
    sensor_msgs::LaserScan bevToLaserScan(const rknpu2::float16* bev_result,
                                         const float transform_array[16],
                                         const bev_utils::BEVConfig& config = bev_utils::BEVConfig(),
                                         const std::string& frame_id = "base_footprint");

    ~SimpleBEV();

private:
    int ret = 0;
    std::mutex mtx;

    ModelPaths model_paths_;
    rknn_context encoder_ctx{}, grid_sample_ctx{}, decoder_ctx{}, lasernet_ctx{};
    unsigned char *encoder_data = nullptr, *grid_sample_data = nullptr, *decoder_data = nullptr, *lasernet_data = nullptr;

    rknn_input_output_num encoder_io_num{}, grid_sample_io_num{}, decoder_io_num{}, lasernet_io_num{};

    InputAttrArrEncoder encoder_input_attrs{};
    InputAttrArrGridSample grid_sample_input_attrs{};
    InputAttrArrDecoder decoder_input_attrs{};
    InputAttrArrLaserNet lasernet_input_attrs{};

    OutputAttrArrEncoder encoder_output_attrs{};
    OutputAttrArrGridSample grid_sample_output_attrs{};
    OutputAttrArrDecoder decoder_output_attrs{};
    OutputAttrArrLaserNet lasernet_output_attrs{};
    
    rknn_tensor_mem* input_img_mems = nullptr;
    rknn_tensor_mem* input_laser_mems = nullptr;

    TensorMemArrEncoder output_mems_encoder{};
    TensorMemArrLaserNet output_mems_lasernet{};
    TensorMemArrGridSample output_mems_grid_sample{};
    TensorMemArrDecoder output_mems_decoder{};

    rknn_tensor_mem* flat_idx_mems = nullptr;

    int channel = 0, width = 0, height = 0;
    int img_width = 0, img_height = 0;
    int input_size = 0;

    float nms_threshold = 0.0f, box_conf_threshold = 0.0f;
};

#endif