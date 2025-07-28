#ifndef SIMPLEBEV_H
#define SIMPLEBEV_H

#include "rknn_api.h"
#include <mutex>
#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include "fp16/Float16.h"

// 常量定义
constexpr int kEncoderInputNum = 1;
constexpr int kGridSampleInputNum = 2;
constexpr int kDecoderInputNum = 1;
constexpr int kEncoderOutputNum = 1;
constexpr int kGridSampleOutputNum = 2;
constexpr int kDecoderOutputNum = 5;

// 类型别名
using TensorAttrArrEncoder = std::array<rknn_tensor_attr, kEncoderInputNum>;
using TensorAttrArrGridSample = std::array<rknn_tensor_attr, kGridSampleInputNum>;
using TensorAttrArrDecoder = std::array<rknn_tensor_attr, kDecoderOutputNum>;
using TensorMemArrEncoder = std::array<rknn_tensor_mem*, kEncoderOutputNum>;
using TensorMemArrGridSample = std::array<rknn_tensor_mem*, kGridSampleOutputNum>;
using TensorMemArrDecoder = std::array<rknn_tensor_mem*, kDecoderOutputNum>;

// 工具函数声明
static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

// SimpleBEV类定义
class SimpleBEV {
public:
    struct ModelPaths {
        std::string encoder_path, grid_sample_path, decoder_path, flat_idx_path;
    };

    explicit SimpleBEV(const ModelPaths& paths);

    int init(rknn_context *encoder_ctx_in,
             rknn_context *grid_sample_ctx_in,
             rknn_context *decoder_ctx_in,
             rknn_tensor_mem* flat_idx_mems_in,
             bool share_weight);

    int init_flat_idx_mems();
    rknn_context* get_encoder_pctx();
    rknn_context* get_grid_sample_pctx();
    rknn_context* get_decoder_pctx();
    rknn_tensor_mem* get_flat_idx_mems();
    int get_input_size() const;

    int query_model_io_num(rknn_context ctx, rknn_input_output_num &io_num, const char* ctx_name);
    int query_input_attributes(rknn_context ctx, rknn_input_output_num& io_num, const char* model_name, rknn_tensor_attr* attrs);
    int query_output_attributes(rknn_context ctx, rknn_input_output_num& io_num, const char* model_name, rknn_tensor_attr* attrs);
    int infer(unsigned char* input_data);
    void visualize_bev_grid(rknpu2::float16* bev_data, int width, int height);

    ~SimpleBEV();

private:
    int ret = 0;
    std::mutex mtx;

    ModelPaths model_paths_;
    rknn_context encoder_ctx{}, grid_sample_ctx{}, decoder_ctx{};
    unsigned char *encoder_data = nullptr, *grid_sample_data = nullptr, *decoder_data = nullptr;

    rknn_input_output_num encoder_io_num{}, grid_sample_io_num{}, decoder_io_num{};
    TensorAttrArrEncoder encoder_input_attrs{};
    TensorAttrArrGridSample grid_sample_input_attrs{};
    TensorAttrArrDecoder decoder_input_attrs{};
    TensorAttrArrEncoder encoder_output_attrs{};
    TensorAttrArrGridSample grid_sample_output_attrs{};
    TensorAttrArrDecoder decoder_output_attrs{};

    rknn_tensor_mem* input_mems = nullptr;
    TensorMemArrEncoder output_mems_encoder{};
    TensorMemArrGridSample output_mems_grid_sample{};
    TensorMemArrDecoder output_mems_decoder{};
    rknn_tensor_mem* flat_idx_mems = nullptr;

    int channel = 0, width = 0, height = 0;
    int img_width = 0, img_height = 0;
    int input_size = 0;

    float nms_threshold = 0.0f, box_conf_threshold = 0.0f;
};

#endif