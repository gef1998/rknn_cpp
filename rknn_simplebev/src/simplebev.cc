#include <stdio.h>
#include <mutex>
#include "rknn_api.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <thread>
#include <future>

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
                    rknn_context *lasernet_ctx_in, 
                    rknn_tensor_mem* flat_idx_mems_in,
                    bool share_weight){
    printf("Loading model...\n");
    int encoder_data_size = 0;
    int grid_sample_data_size = 0;
    int decoder_data_size = 0;
    int lasernet_data_size = 0;

    encoder_data = load_model(model_paths_.encoder_path.c_str(), &encoder_data_size);
    grid_sample_data = load_model(model_paths_.grid_sample_path.c_str(), &grid_sample_data_size);
    decoder_data = load_model(model_paths_.decoder_path.c_str(), &decoder_data_size);
    lasernet_data = load_model(model_paths_.lasernet_path.c_str(), &lasernet_data_size);

    // 模型参数复用/Model parameter reuse
    if (share_weight == true){
        ret = rknn_dup_context(encoder_ctx_in, &encoder_ctx);
        ret = rknn_dup_context(grid_sample_ctx_in, &grid_sample_ctx);
        ret = rknn_dup_context(decoder_ctx_in, &decoder_ctx);
        ret = rknn_dup_context(lasernet_ctx_in, &lasernet_ctx);
    }
    else{
        ret = rknn_init(&encoder_ctx, encoder_data, encoder_data_size, 0, NULL);
        ret = rknn_init(&grid_sample_ctx, grid_sample_data, grid_sample_data_size, 0, NULL);
        ret = rknn_init(&decoder_ctx, decoder_data, decoder_data_size, 0, NULL);
        ret = rknn_init(&lasernet_ctx, lasernet_data, lasernet_data_size, 0, NULL);
        }

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
    ret = rknn_set_core_mask(lasernet_ctx, core_mask);

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
    query_model_io_num(lasernet_ctx, lasernet_io_num, "lasernet");

    query_input_attributes(encoder_ctx, encoder_io_num, "encoder", encoder_input_attrs.data());
    query_input_attributes(grid_sample_ctx, grid_sample_io_num, "grid_sample", grid_sample_input_attrs.data());
    query_input_attributes(decoder_ctx, decoder_io_num, "decoder", decoder_input_attrs.data());
    query_input_attributes(lasernet_ctx, lasernet_io_num, "lasernet", lasernet_input_attrs.data());

    query_output_attributes(encoder_ctx, encoder_io_num, "encoder", encoder_output_attrs.data());
    query_output_attributes(grid_sample_ctx, grid_sample_io_num, "grid_sample", grid_sample_output_attrs.data());
    query_output_attributes(decoder_ctx, decoder_io_num, "decoder", decoder_output_attrs.data());
    query_output_attributes(lasernet_ctx, lasernet_io_num, "lasernet", lasernet_output_attrs.data());

    if (share_weight == true){
        this->flat_idx_mems = flat_idx_mems_in;
    }
    else{
        this->init_flat_idx_mems();
    }

    input_img_mems = rknn_create_mem(encoder_ctx, encoder_input_attrs[0].size_with_stride);
    // Copy input data to input tensor memory
    int width  = encoder_input_attrs[0].dims[2];
    int stride = encoder_input_attrs[0].w_stride;
    if (width != stride) {
        printf("width != stride");
        return -1;}
        // if (width == stride) {
        //   memcpy(input_img_mems->virt_addr, input_data, encoder_input_attrs[0].size);
        // } else { 
        //     printf("[ERROR!] width != stride");
        //   }
    encoder_input_attrs[0].type = (rknn_tensor_type)RKNN_TENSOR_UINT8;
    encoder_input_attrs[0].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
    ret = rknn_set_io_mem(encoder_ctx, input_img_mems, &encoder_input_attrs[0]);

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

    input_laser_mems = rknn_create_mem(lasernet_ctx, lasernet_input_attrs[0].size_with_stride);
    // Copy input data to input tensor memory
    width  = lasernet_input_attrs[0].dims[2];
    stride = lasernet_input_attrs[0].w_stride;
    printf("width = %d, stride = %d\n", width, stride);
    if (width != stride) {
        printf("width != stride");
        return -1;}
        // if (width == stride) {
        //   memcpy(input_img_mems->virt_addr, input_data, encoder_input_attrs[0].size);
        // } else { 
        //     printf("[ERROR!] width != stride");
        //   }
    lasernet_input_attrs[0].type = (rknn_tensor_type)RKNN_TENSOR_FLOAT16;
    lasernet_input_attrs[0].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
    ret = rknn_set_io_mem(lasernet_ctx, input_laser_mems, &lasernet_input_attrs[0]);
    for (uint32_t i = 0; i < lasernet_io_num.n_output; ++i) {
        int output_size = lasernet_output_attrs[i].n_elems * 2;
        this->output_mems_lasernet[i]  = rknn_create_mem(lasernet_ctx, output_size);
      }

    // Set output tensor memory
    for (uint32_t i = 0; i < lasernet_io_num.n_output; ++i) {
        lasernet_output_attrs[i].type = RKNN_TENSOR_FLOAT16;
        lasernet_output_attrs[i].fmt = RKNN_TENSOR_NHWC;
        ret = rknn_set_io_mem(lasernet_ctx, output_mems_lasernet[i], &lasernet_output_attrs[i]);
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
    decoder_input_attrs[1].type = (rknn_tensor_type)RKNN_TENSOR_FLOAT16;
    decoder_input_attrs[1].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;

    // Set input tensor memory
    ret = rknn_set_io_mem(decoder_ctx, output_mems_grid_sample[0], &decoder_input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;}
    ret = rknn_set_io_mem(decoder_ctx, output_mems_lasernet[0], &decoder_input_attrs[1]);
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

rknn_context *SimpleBEV::get_lasernet_pctx()
{
    return &lasernet_ctx;
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

// rknpu2::float16 * SimpleBEV::infer(unsigned char* input_data)
// {
//     std::lock_guard<std::mutex> lock(mtx);
//     memcpy(input_img_mems->virt_addr, input_data, encoder_input_attrs[0].size);
//     // 模型推理/Model inference
//     ret = rknn_run(encoder_ctx, NULL);
//     ret = rknn_run(grid_sample_ctx, NULL);
//     if (ret < 0) {
//       printf("grid_sample rknn_run fail! ret=%d\n", ret);
//       return nullptr;
//     }
//     ret = rknn_run(decoder_ctx, NULL);
//     if (ret < 0) {
//       printf("decoder rknn_run fail! ret=%d\n", ret);
//       return nullptr;
//     }


//     // for(int i=0; i<encoder_io_num.n_output; i++){
        
//     //     printf("encoder output[%d] shape: [", i);
//     //     for(int j=0; j<encoder_output_attrs[i].n_dims; j++){
//     //         printf("%d ", encoder_output_attrs[i].dims[j]);
//     //     }
//     //     printf("]\n");
        
//     //     auto* out_data = (rknpu2::float16 *)this->output_mems_encoder[i]->virt_addr;
//     //     for(int k=0; k<10; k++){ // 示例只打印前10个数据
//     //         printf("%f ", (float)out_data[k]);
//     //     }
//     //     printf("\n...\n");
//     // }

//     //     auto* out_data1 = (int8_t *)output_mems_grid_sample[0]->virt_addr;
//     //     for(int k=0; k<10; k++){ // 示例只打印前10个数据
//     //         printf("%d ", out_data1[k]);
//     //     }
//     //     printf("\n...\n");

//     //     auto* out_data2 = (rknpu2::float16 *)output_mems_grid_sample[1]->virt_addr;
//     //     for(int k=0; k<10; k++){ // 示例只打印前10个数据
//     //         printf("%f ", (float)out_data2[k]);
//     //     }
//     //     printf("\n...\n");

//         for(int i=0; i<decoder_io_num.n_output; i++){
//             printf("decoder output[%d] data: ", i);
//             auto* out_data_decoder = (rknpu2::float16 *)output_mems_decoder[i]->virt_addr;
//             for(int k=0; k<10; k++){ // 示例只打印前10个数据
//                 printf("%f ", (float)out_data_decoder[k]);
//             }
//             printf("\n...\n");
//         }
        
//         // 对decoder的第2个输出进行BEV可视化 (96x96的可行驶区域分割结果)
//         // auto* out_data_decoder = (rknpu2::float16 *)output_mems_decoder[2]->virt_addr;
//         // visualize_bev_grid(out_data_decoder, 96, 96);

//     return (rknpu2::float16 *)output_mems_decoder[2]->virt_addr;
// }

rknpu2::float16 * SimpleBEV::infer_multi_sensor(unsigned char* image_data, rknpu2::float16* pointcloud_data)
{
    memcpy(input_img_mems->virt_addr, image_data, encoder_input_attrs[0].size);
    memcpy(input_laser_mems->virt_addr, pointcloud_data, lasernet_input_attrs[0].size);

    // rknpu2::float16* arr = new rknpu2::float16[73728];
    // std::fill(arr, arr+73728, rknpu2::float16(-0.5f));
    // memcpy(output_mems_lasernet[0]->virt_addr, arr, lasernet_output_attrs[0].size);

    // 模型推理/Model inference
    // 并行执行两个模型推理
    std::future<int> lasernet_future = std::async(std::launch::async, [this]() {
        return rknn_run(this->lasernet_ctx, NULL);
    });

    std::future<int> encoder_future = std::async(std::launch::async, [this]() {
        return rknn_run(this->encoder_ctx, NULL);
    });

    // 等待两个推理完成
    int lasernet_ret = lasernet_future.get();
    int encoder_ret = encoder_future.get();

    if (encoder_ret < 0 || lasernet_ret < 0) {
        printf("Parallel inference failed! encoder_ret=%d, grid_sample_ret=%d\n", 
                encoder_ret, lasernet_ret);
        return nullptr;
    }

    // ret = rknn_run(lasernet_ctx, NULL);
    // ret = rknn_run(encoder_ctx, NULL);
    ret = rknn_run(grid_sample_ctx, NULL);
    if (ret < 0) {
      printf("grid_sample rknn_run fail! ret=%d\n", ret);
      return nullptr;
    }
    ret = rknn_run(decoder_ctx, NULL);
    if (ret < 0) {
      printf("decoder rknn_run fail! ret=%d\n", ret);
      return nullptr;
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

        // for(int i=0; i<decoder_io_num.n_output; i++){
        //     printf("decoder output[%d] data: ", i);
        //     auto* out_data_decoder = (rknpu2::float16 *)output_mems_decoder[i]->virt_addr;
        //     for(int k=0; k<10; k++){ // 示例只打印前10个数据
        //         printf("%f ", (float)out_data_decoder[k]);
        //     }
        //     printf("\n...\n");
        // }

        // auto* out_data_lasernet = (rknpu2::float16 *)output_mems_lasernet[0]->virt_addr;
        // float sum = 0.0f;
        // for(int k=0; k<73728; k++){ // 示例只打印前10个数据
        //     sum += (float)out_data_lasernet[k] + 0.5f;
        //     // printf("%f ", (float)out_data_lasernet[k]);
        // }
        // printf("lasernet output sum: %f\n", sum);
        // printf("\n...\n");

    return (rknpu2::float16 *)output_mems_decoder[0]->virt_addr;
}

// 重载的infer方法，使用MultiSensorData结构
rknpu2::float16* SimpleBEV::infer(const MultiSensorData& sensor_data) {
    return infer_multi_sensor(sensor_data.image_data, sensor_data.pointcloud_data);
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
