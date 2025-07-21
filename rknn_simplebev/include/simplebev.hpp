#ifndef SIMPLEBEV_H
#define SIMPLEBEV_H

#include "rknn_api.h"
#include <mutex> 

#define RGB_CAMXS_SIZE 8*3*240*320
#define FLAT_IDX_ATTR grid_sample_input_attrs[1]
#define RGB_CAMXS_ATTR encoder_input_attrs[0]
 
static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);


// struct ImgBbox {
//     cv::Mat image;
//     detect_result_group_t detection;
// };

class SimpleBEV
{
private:
    int ret;
    std::mutex mtx;

    std::string encoder_path, grid_sample_path, decoder_path, flat_idx_path;
    rknn_context encoder_ctx, grid_sample_ctx, decoder_ctx;

    unsigned char *encoder_data, *grid_sample_data, *decoder_data;
    rknn_input_output_num encoder_io_num, grid_sample_io_num, decoder_io_num;
    rknn_tensor_attr encoder_input_attrs[1];
    rknn_tensor_attr grid_sample_input_attrs[2];
    rknn_tensor_attr decoder_input_attrs[1];

    rknn_tensor_attr encoder_output_attrs[1];
    rknn_tensor_attr grid_sample_output_attrs[2];
    rknn_tensor_attr decoder_output_attrs[5];
    
    rknn_tensor_mem* input_mems;
    rknn_tensor_mem *output_mems_encoder[1];
    rknn_tensor_mem *output_mems_grid_sample[2];
    rknn_tensor_mem *output_mems_decoder[5];
    rknn_tensor_mem* flat_idx_mems;

    int channel, width, height;
    int img_width, img_height;
    int input_size;

    float nms_threshold, box_conf_threshold;

public:
    SimpleBEV(const std::string &encoder_path,
            const std::string &preprosess_path,
            const std::string &grid_sample_path,
            const std::string &decoder_path);
    int init(rknn_context *encoder_ctx_in, 
            rknn_context *grid_sample_ctx_in, 
            rknn_context *decoder_ctx_in, 
            rknn_tensor_mem* flat_idx_mems_in,
            bool share_weight);
    int init_flat_idx_mems();
    rknn_context *get_encoder_pctx();
    rknn_context *get_grid_sample_pctx();
    rknn_context *get_decoder_pctx();
    rknn_tensor_mem *get_flat_idx_mems();
    int get_input_size();

    int query_model_io_num(rknn_context ctx, rknn_input_output_num &io_num, const char* ctx_name);
    int query_input_attributes(rknn_context ctx, rknn_input_output_num& io_num, const char* model_name, rknn_tensor_attr* attrs);
    int query_output_attributes(rknn_context ctx, rknn_input_output_num& io_num, const char* model_name, rknn_tensor_attr* attrs);
    int infer(unsigned char* input_data);
    ~SimpleBEV();
};

#endif