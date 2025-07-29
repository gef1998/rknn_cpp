#include <stdio.h>
#include <mutex>
#include "rknn_api.h"
#include <iostream>
#include <cstring>
#include <cmath>

#include "coreNum.hpp"
#include "simplebev.hpp"
#include "fp16/Float16.h"

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

SimpleBEV::SimpleBEV(const ModelPaths& paths)
    : model_paths_(paths) {}

int SimpleBEV::init(rknn_context *encoder_ctx_in, 
                    rknn_context *grid_sample_ctx_in, 
                    rknn_context *decoder_ctx_in, 
                    rknn_tensor_mem* flat_idx_mems_in,
                    bool share_weight){
    printf("Loading model...\n");
    int encoder_data_size = 0;
    int grid_sample_data_size = 0;
    int decoder_data_size = 0;

    encoder_data = load_model(model_paths_.encoder_path.c_str(), &encoder_data_size);
    grid_sample_data = load_model(model_paths_.grid_sample_path.c_str(), &grid_sample_data_size);
    decoder_data = load_model(model_paths_.decoder_path.c_str(), &decoder_data_size);

    // 模型参数复用/Model parameter reuse
    if (share_weight == true){
        ret = rknn_dup_context(encoder_ctx_in, &encoder_ctx);
        ret = rknn_dup_context(grid_sample_ctx_in, &grid_sample_ctx);
        ret = rknn_dup_context(decoder_ctx_in, &decoder_ctx);}
    else{
        ret = rknn_init(&encoder_ctx, encoder_data, encoder_data_size, 0, NULL);
        ret = rknn_init(&grid_sample_ctx, grid_sample_data, grid_sample_data_size, 0, NULL);
        ret = rknn_init(&decoder_ctx, decoder_data, decoder_data_size, 0, NULL);}
        
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(encoder_ctx, core_mask);
    ret = rknn_set_core_mask(grid_sample_ctx, core_mask);
    ret = rknn_set_core_mask(decoder_ctx, core_mask);

    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(encoder_ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // 获取模型输入输出参数/Obtain the input and output parameters of the model
    query_model_io_num(encoder_ctx, encoder_io_num, "encoder");
    query_model_io_num(grid_sample_ctx, grid_sample_io_num, "grid_sample");
    query_model_io_num(decoder_ctx, decoder_io_num, "decoder");

    query_input_attributes(encoder_ctx, encoder_io_num, "encoder", encoder_input_attrs.data());
    query_input_attributes(grid_sample_ctx, grid_sample_io_num, "grid_sample", grid_sample_input_attrs.data());
    query_input_attributes(decoder_ctx, decoder_io_num, "decoder", decoder_input_attrs.data());

    query_output_attributes(encoder_ctx, encoder_io_num, "encoder", encoder_output_attrs.data());
    query_output_attributes(grid_sample_ctx, grid_sample_io_num, "grid_sample", grid_sample_output_attrs.data());
    query_output_attributes(decoder_ctx, decoder_io_num, "decoder", decoder_output_attrs.data());

    if (share_weight == true){
        this->flat_idx_mems = flat_idx_mems_in;
    }
    else{
        this->init_flat_idx_mems();
    }
    input_size = encoder_input_attrs[0].size;
    input_mems = rknn_create_mem(grid_sample_ctx, encoder_input_attrs[0].size_with_stride);
        // Copy input data to input tensor memory
    int width  = encoder_input_attrs[0].dims[2];
    int stride = encoder_input_attrs[0].w_stride;
    if (width != stride) {
        printf("width != stride");
        return -1;}
        // if (width == stride) {
        //   memcpy(input_mems->virt_addr, input_data, encoder_input_attrs[0].size);
        // } else { 
        //     printf("[ERROR!] width != stride");
        //   }
    encoder_input_attrs[0].type = (rknn_tensor_type)RKNN_TENSOR_UINT8;
    encoder_input_attrs[0].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
    ret = rknn_set_io_mem(encoder_ctx, input_mems, &encoder_input_attrs[0]);

    for (uint32_t i = 0; i < encoder_io_num.n_output; ++i) {
      int output_size = encoder_output_attrs[i].n_elems * 2;
      this->output_mems_encoder[i]  = rknn_create_mem(encoder_ctx, output_size);
    }
    // Set output tensor memory
    for (uint32_t i = 0; i < encoder_io_num.n_output; ++i) {
        encoder_output_attrs[i].type = RKNN_TENSOR_FLOAT16;
        encoder_output_attrs[i].fmt = RKNN_TENSOR_NHWC;
        ret = rknn_set_io_mem(encoder_ctx, output_mems_encoder[i], &encoder_output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }
    grid_sample_input_attrs[0].type = (rknn_tensor_type)RKNN_TENSOR_FLOAT16;
    grid_sample_input_attrs[0].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
    grid_sample_input_attrs[1].type = (rknn_tensor_type)RKNN_TENSOR_FLOAT16;
    grid_sample_input_attrs[1].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
    // Set input tensor memory
    ret = rknn_set_io_mem(grid_sample_ctx, output_mems_encoder[0], &grid_sample_input_attrs[0]);
    if (ret < 0) {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
    }
    ret = rknn_set_io_mem(grid_sample_ctx, flat_idx_mems, &grid_sample_input_attrs[1]);
    
    int bs_array_grid_sample[2] = {2, 2};
    // Allocate output memory
    for (uint32_t i = 0; i < grid_sample_io_num.n_output; ++i) {
      int output_size = grid_sample_output_attrs[i].n_elems * bs_array_grid_sample[i];
      output_mems_grid_sample[i]  = rknn_create_mem(grid_sample_ctx, output_size);
    }

    grid_sample_output_attrs[0].type = RKNN_TENSOR_FLOAT16;
    grid_sample_output_attrs[0].fmt = RKNN_TENSOR_NHWC;      
    grid_sample_output_attrs[1].type = RKNN_TENSOR_FLOAT16;
    grid_sample_output_attrs[1].fmt = RKNN_TENSOR_UNDEFINED;  
  
    // Set output tensor memory
    for (uint32_t i = 0; i < grid_sample_io_num.n_output; ++i) {
    // set output memory and attribute
        ret = rknn_set_io_mem(grid_sample_ctx, output_mems_grid_sample[i], &grid_sample_output_attrs[i]);
    if (ret < 0) {
        printf("grid_sample rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;}
    }

    decoder_input_attrs[0].type = (rknn_tensor_type)RKNN_TENSOR_FLOAT16;
    decoder_input_attrs[0].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;

    // Set input tensor memory
    ret = rknn_set_io_mem(decoder_ctx, output_mems_grid_sample[0], &decoder_input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;}

    // Allocate output memory
    for (uint32_t i = 0; i < decoder_io_num.n_output; ++i) {
        int output_size = decoder_output_attrs[i].n_elems * 2;
        output_mems_decoder[i]  = rknn_create_mem(decoder_ctx, output_size);
    }

    decoder_output_attrs[0].type = RKNN_TENSOR_FLOAT16;
    decoder_output_attrs[0].fmt = RKNN_TENSOR_NCHW;    
    decoder_output_attrs[1].type = RKNN_TENSOR_FLOAT16;
    decoder_output_attrs[1].fmt = RKNN_TENSOR_NCHW;  
    decoder_output_attrs[2].type = RKNN_TENSOR_FLOAT16;
    decoder_output_attrs[2].fmt = RKNN_TENSOR_NCHW;  
    decoder_output_attrs[3].type = RKNN_TENSOR_FLOAT16;
    decoder_output_attrs[3].fmt = RKNN_TENSOR_NCHW;  
    decoder_output_attrs[4].type = RKNN_TENSOR_FLOAT16;
    decoder_output_attrs[4].fmt = RKNN_TENSOR_NCHW;

    // Set output tensor memory
    for (uint32_t i = 0; i < decoder_io_num.n_output; ++i) {
    // set output memory and attribute
        ret = rknn_set_io_mem(decoder_ctx, output_mems_decoder[i], &decoder_output_attrs[i]);
    if (ret < 0) {
        printf("decoder rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }
    }
    return 0;
}

rknn_context *SimpleBEV::get_encoder_pctx()
{
    return &encoder_ctx;
}

rknn_context *SimpleBEV::get_grid_sample_pctx()
{
    return &grid_sample_ctx;
}

rknn_context *SimpleBEV::get_decoder_pctx()
{
    return &decoder_ctx;
}

rknn_tensor_mem *SimpleBEV::get_flat_idx_mems()
{
    return flat_idx_mems;
}


static int file_counter = 1;  // 全局文件计数器

void save_float16_to_bin(rknpu2::float16* data, size_t count) {
    // 生成自增文件名
    char filename[32];
    snprintf(filename, sizeof(filename), "%d.bin", file_counter++);
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file");
        return;
    }
        
    // 写入实际数据
    fwrite(data, sizeof(rknpu2::float16), count, fp);
    
    fclose(fp);
    printf("Saved to %s\n", filename);
}

int SimpleBEV::infer(unsigned char* input_data)
{
    std::lock_guard<std::mutex> lock(mtx);
    memcpy(input_mems->virt_addr, input_data, encoder_input_attrs[0].size);
    // 模型推理/Model inference
    ret = rknn_run(encoder_ctx, NULL);
    ret = rknn_run(grid_sample_ctx, NULL);
    if (ret < 0) {
      printf("grid_sample rknn_run fail! ret=%d\n", ret);
      return -1;
    }
    ret = rknn_run(decoder_ctx, NULL);
    if (ret < 0) {
      printf("decoder rknn_run fail! ret=%d\n", ret);
      return -1;
    }


    // for(int i=0; i<encoder_io_num.n_output; i++){
        
    //     printf("encoder output[%d] shape: [", i);
    //     for(int j=0; j<encoder_output_attrs[i].n_dims; j++){
    //         printf("%d ", encoder_output_attrs[i].dims[j]);
    //     }
    //     printf("]\n");
        
    //     auto* out_data = (rknpu2::float16 *)this->output_mems_encoder[i]->virt_addr;
    //     for(int k=0; k<10; k++){ // 示例只打印前10个数据
    //         printf("%f ", (float)out_data[k]);
    //     }
    //     printf("\n...\n");
    // }

    //     auto* out_data1 = (int8_t *)output_mems_grid_sample[0]->virt_addr;
    //     for(int k=0; k<10; k++){ // 示例只打印前10个数据
    //         printf("%d ", out_data1[k]);
    //     }
    //     printf("\n...\n");

    //     auto* out_data2 = (rknpu2::float16 *)output_mems_grid_sample[1]->virt_addr;
    //     for(int k=0; k<10; k++){ // 示例只打印前10个数据
    //         printf("%f ", (float)out_data2[k]);
    //     }
    //     printf("\n...\n");

        for(int i=0; i<decoder_io_num.n_output; i++){
            printf("decoder output[%d] data: ", i);
            auto* out_data_decoder = (rknpu2::float16 *)output_mems_decoder[i]->virt_addr;
            for(int k=0; k<10; k++){ // 示例只打印前10个数据
                printf("%f ", (float)out_data_decoder[k]);
            }
            printf("\n...\n");
        }
        
        // 对decoder的第2个输出进行BEV可视化 (96x96的可行驶区域分割结果)
        // auto* out_data_decoder = (rknpu2::float16 *)output_mems_decoder[2]->virt_addr;
        // visualize_bev_grid(out_data_decoder, 96, 96);

    return 1;
}

int SimpleBEV::init_flat_idx_mems() {
    if (model_paths_.flat_idx_path.empty()) {
        std::cerr << "Error: flat_idx_path is empty!" << std::endl;
        return -1;
    }

    unsigned char* flat_idx_data;
    // Load input
    flat_idx_data = new unsigned char[grid_sample_input_attrs[1].size]; 

    printf("%s\n", model_paths_.flat_idx_path.c_str());
    FILE* fp_flat_idx = fopen(model_paths_.flat_idx_path.c_str(), "rb");
    if (fp_flat_idx == NULL) {
      perror("open failed!");
      return -1;
    }

    fread(flat_idx_data, grid_sample_input_attrs[1].size, 1, fp_flat_idx);
    fclose(fp_flat_idx);
    this->flat_idx_mems = rknn_create_mem(grid_sample_ctx, grid_sample_input_attrs[1].size_with_stride);
    // Copy input data to input tensor memory
    const int width  = grid_sample_input_attrs[1].dims[2];
    const int stride = grid_sample_input_attrs[1].w_stride;
    if (width == stride) {
      memcpy(this->flat_idx_mems->virt_addr, flat_idx_data, grid_sample_input_attrs[1].size);
    } else { 
        std::cerr << "Critical error: width (" << width  << ") != stride (" << stride << ")" << std::endl;
        rknn_destroy_mem(grid_sample_ctx, flat_idx_mems);
        flat_idx_mems = nullptr;
        return -1;
    }
    delete flat_idx_data;
    return 0;
  }

int SimpleBEV::query_model_io_num(rknn_context ctx, rknn_input_output_num &io_num, const char* ctx_name) {
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("%s rknn_query fail! ret=%d\n", ctx_name, ret);
        return -1;
    }
    printf("[%s] input num: %d, output num: %d\n", 
           ctx_name, io_num.n_input, io_num.n_output);
    return 0;
  }

int SimpleBEV::query_input_attributes(rknn_context ctx, 
                                    rknn_input_output_num& io_num,
                                    const char* model_name,
                                    rknn_tensor_attr* attrs) {
    printf("%s input tensors:\n", model_name);
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        attrs[i].index = i;
        int ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("%s rknn_query fail! ret=%d\n", model_name, ret);
            return -1;
        }
        dump_tensor_attr(&attrs[i]);
    }
    return 0;
}

int SimpleBEV::query_output_attributes(rknn_context ctx, 
                                    rknn_input_output_num& io_num,
                                    const char* model_name,
                                    rknn_tensor_attr* attrs) {
    printf("%s output tensors:\n", model_name);
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        attrs[i].index = i;
        int ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("%s rknn_query fail! ret=%d\n", model_name, ret);
        return -1;
        }
        dump_tensor_attr(&attrs[i]);
    }
    return 0;
    }

int SimpleBEV::get_input_size() const {
    return input_size;
}

void SimpleBEV::visualize_bev_grid(rknpu2::float16* bev_data, int width, int height) {
    // 创建OpenCV矩阵
    cv::Mat bev_image(height, width, CV_8UC3);
    
    // 添加数据调试信息
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    float sigmoid_min = FLT_MAX, sigmoid_max = -FLT_MAX;

    // 对每个像素进行sigmoid处理和二值化
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;            
            float raw_val = static_cast<float>(bev_data[idx]);                        
            // Sigmoid激活函数: 1 / (1 + exp(-x))
            float sigmoid_val = 1.0f / (1.0f + std::exp(-raw_val));                        
            // 二值化：>0.5为可行驶区域(白色)，<=0.5为不可行驶区域(黑色)
            uchar pixel_val = (sigmoid_val > 0.5f) ? 255 : 0;            
            // 设置RGB值
            bev_image.at<cv::Vec3b>(y, x) = cv::Vec3b(pixel_val, pixel_val, pixel_val);
        }
    }
    bev_image.at<cv::Vec3b>(47, 47) = cv::Vec3b(0, 0, 255);
    bev_image.at<cv::Vec3b>(48, 48) = cv::Vec3b(0, 0, 255);
    bev_image.at<cv::Vec3b>(48, 47) = cv::Vec3b(0, 0, 255);
    bev_image.at<cv::Vec3b>(47, 48) = cv::Vec3b(0, 0, 255);
    
    // 放大显示图像，便于观察细节 (96x96 -> 480x480)
    // cv::Mat enlarged_image;
    // cv::resize(bev_image, enlarged_image, cv::Size(480, 480), 0, 0, cv::INTER_NEAREST);

    // 直接显示图像
    cv::imshow("BEV Inderence", bev_image);
    cv::waitKey(1); // 非阻塞显示，允许实时更新
    printf("BEV网格已显示 (白色=可行驶，黑色=不可行驶)\n");
}

SimpleBEV::~SimpleBEV()
{
    rknn_destroy(encoder_ctx);
    rknn_destroy(grid_sample_ctx);
    rknn_destroy(decoder_ctx);

    if (encoder_data)
        free(encoder_data);
    if (grid_sample_data)
        free(grid_sample_data);
    if (decoder_data)
        free(decoder_data);

    for (size_t i = 0; i < output_mems_encoder.size(); ++i) {
        if (output_mems_encoder[i])
            rknn_destroy_mem(encoder_ctx, output_mems_encoder[i]);
    }
    for (size_t i = 0; i < output_mems_grid_sample.size(); ++i) {
        if (output_mems_grid_sample[i])
            rknn_destroy_mem(grid_sample_ctx, output_mems_grid_sample[i]);
    }
    for (size_t i = 0; i < output_mems_decoder.size(); ++i) {
        if (output_mems_decoder[i])
            rknn_destroy_mem(decoder_ctx, output_mems_decoder[i]);
    }
}
