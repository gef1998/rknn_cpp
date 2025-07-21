// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "rknn_api.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include "fp16/Float16.h"
#include <chrono>

#define TOTAL_RKNN_MODEL_NUM 2

/*-------------------------------------------
                  Functions
-------------------------------------------*/

void* thread_func(void* ctx) {
  rknn_context* model_ctx = (rknn_context*)ctx;
  int ret = rknn_run(*model_ctx, NULL);
  if (ret < 0) {
      printf("rknn_run fail! ret=%d\n", ret);
      pthread_exit((void*)-1);
  }
  return NULL;
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static std::vector<std::string> split(const std::string& str, const std::string& pattern)
{
  std::vector<std::string> res;
  if (str == "")
    return res;
  std::string strs = str + pattern;
  size_t      pos  = strs.find(pattern);
  while (pos != strs.npos) {
    std::string temp = strs.substr(0, pos);
    res.push_back(temp);
    strs = strs.substr(pos + 1, strs.size());
    pos  = strs.find(pattern);
  }
  return res;
}

int init_rknn_model(rknn_context* ctx, char* model_path, const char* ctx_name) {
  printf("\033[0;32mLoading %s ... \033[0;0m\n", model_path);
  int ret = rknn_init(ctx, model_path, 0, 0, NULL);
  if (ret < 0) {
      printf("%s rknn_init fail! ret=%d\n", ctx_name, ret);
  }
  return ret;
}

int query_model_io_num(rknn_context ctx, rknn_input_output_num &io_num, const char* ctx_name) {
  int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
      printf("%s rknn_query fail! ret=%d\n", ctx_name, ret);
      return -1;
  }
  printf("[%s] input num: %d, output num: %d\n", 
         ctx_name, io_num.n_input, io_num.n_output);
  return 0;
}

/**
 * 查询模型输入张量属性
 * @param ctx 模型上下文
 * @param io_num 输入输出数量结构体
 * @param model_name 模型名称(用于日志)
 * @param attrs 输出属性数组(需预先分配足够空间)
 * @return 成功返回0，失败返回-1
 */
int query_input_attributes(rknn_context ctx, 
                          rknn_input_output_num* io_num,
                          const char* model_name,
                          rknn_tensor_attr* attrs) {
  printf("%s input tensors:\n", model_name);
  memset(attrs, 0, io_num->n_input * sizeof(rknn_tensor_attr));

  for (uint32_t i = 0; i < io_num->n_input; i++) {
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


/**
 * 查询模型输出张量属性
 * @param ctx 模型上下文
 * @param io_num 输入输出数量结构体
 * @param model_name 模型标识(用于日志)
 * @param attrs 输出属性数组指针(需预分配空间)
 * @return 成功返回0，失败返回-1
 */
int query_output_attributes(rknn_context ctx, 
                          rknn_input_output_num* io_num,
                          const char* model_name,
                          rknn_tensor_attr* attrs) {
  printf("%s output tensors:\n", model_name);
  memset(attrs, 0, io_num->n_output * sizeof(rknn_tensor_attr));

  for (uint32_t i = 0; i < io_num->n_output; i++) {
    attrs[i].index = i;
    int ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR,
      &attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("%s rknn_query fail! ret=%d\n", model_name, ret);
      return -1;
    }
    dump_tensor_attr(&attrs[i]);
  }
  return 0;
}

/**
 * 查询模型自定义字符串
 * @param ctx 模型上下文
 * @param model_name 模型标识(用于日志)
 * @param custom_str 输出字符串结构体指针
 * @return 成功返回0，失败返回-1
 */
int query_custom_string(rknn_context ctx, 
          const char* model_name,
          rknn_custom_string* custom_str) {
  memset(custom_str, 0, sizeof(rknn_custom_string));
  int ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, 
    custom_str, sizeof(rknn_custom_string));
  if (ret != RKNN_SUCC) {
    printf("%s rknn_query fail! ret=%d\n", model_name, ret);
    return -1;
    }
  printf("%s custom string: %s\n", model_name, custom_str->string);
  return 0;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char* argv[]){
  char* encoder_path = argv[1];
  char* encoder_input_paths = argv[2];
  char* preprosess_path = argv[3];
  char* preprosess_input_paths = argv[4];
  char* grid_sample_path = argv[5];
  char* decoder_path = argv[6];

  std::vector<std::string> encoder_input_paths_split = split(encoder_input_paths, "#");
  std::vector<std::string> preprosess_input_paths_split = split(preprosess_input_paths, "#");

  rknn_context ctx_encoder, ctx_preprosess, ctx_grid_sample, ctx_decoder;
  // Get sdk and driver version

  if (init_rknn_model(&ctx_encoder, encoder_path, "ctx_encoder") < 0 ||
      init_rknn_model(&ctx_preprosess, preprosess_path, "ctx_preprosess") < 0 ||
      init_rknn_model(&ctx_grid_sample, grid_sample_path, "ctx_grid_sample") < 0 ||
      init_rknn_model(&ctx_decoder, decoder_path, "ctx_decoder") < 0) {
  return -1; }

  rknn_sdk_version sdk_ver;
  int ret = rknn_query(ctx_encoder, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);
    
  rknn_input_output_num io_num_encoder, io_num_preprosess, io_num_grid_sample, io_num_decoder;
  if (query_model_io_num(ctx_encoder, io_num_encoder, "encoder") < 0 ||
      query_model_io_num(ctx_preprosess, io_num_preprosess, "preprocess") < 0 ||
      query_model_io_num(ctx_grid_sample, io_num_grid_sample, "grid_sample") < 0 ||
      query_model_io_num(ctx_decoder, io_num_decoder, "decoder") < 0) {
  return -1;  // 任一失败则整体退出
  }

  rknn_tensor_attr input_attrs_encoder[io_num_encoder.n_input];
  rknn_tensor_attr input_attrs_preprosess[io_num_preprosess.n_input];
  rknn_tensor_attr input_attrs_grid_sample[io_num_grid_sample.n_input];
  rknn_tensor_attr input_attrs_decoder[io_num_decoder.n_input];


  if (query_input_attributes(ctx_encoder, &io_num_encoder, "encoder", input_attrs_encoder) < 0 ||
      query_input_attributes(ctx_preprosess, &io_num_preprosess, "preprocess", input_attrs_preprosess) < 0 ||
      query_input_attributes(ctx_grid_sample, &io_num_grid_sample, "grid_sample", input_attrs_grid_sample) < 0 ||
      query_input_attributes(ctx_decoder, &io_num_decoder, "decoder", input_attrs_decoder) < 0) {
      return -1;
  }

  // 声明输出属性数组
  rknn_tensor_attr output_attrs_encoder[io_num_encoder.n_output];
  rknn_tensor_attr output_attrs_preprosess[io_num_preprosess.n_output];
  rknn_tensor_attr output_attrs_grid_sample[io_num_grid_sample.n_output];
  rknn_tensor_attr output_attrs_decoder[io_num_decoder.n_output];

// 统一查询输出属性
  if (query_output_attributes(ctx_encoder, &io_num_encoder, "encoder", output_attrs_encoder) < 0 ||
      query_output_attributes(ctx_preprosess, &io_num_preprosess, "preprocess", output_attrs_preprosess) < 0 ||
      query_output_attributes(ctx_grid_sample, &io_num_grid_sample, "grid_sample", output_attrs_grid_sample) < 0 ||
      query_output_attributes(ctx_decoder, &io_num_decoder, "decoder", output_attrs_decoder) < 0) {
  return -1;
  }

  rknn_custom_string encoder_str, preprocess_str, grid_sample_str, decoder_str;
  if (query_custom_string(ctx_encoder, "encoder", &encoder_str) < 0 ||
      query_custom_string(ctx_preprosess, "preprocess", &preprocess_str) < 0 ||
      query_custom_string(ctx_grid_sample, "grid_sample", &grid_sample_str) < 0 ||
      query_custom_string(ctx_decoder, "decoder", &decoder_str) < 0) {
  return -1;
  }
  
  
// =============init encoder========================
  unsigned char* input_data = nullptr;
  
  // Load input
  if (io_num_encoder.n_input != encoder_input_paths_split.size()) {
    printf("input number mismatch! model input num: %d, input paths num: %zu\n", io_num_encoder.n_input, encoder_input_paths_split.size());
    return -1;
  }
  input_data = new unsigned char[input_attrs_encoder[0].size];
  printf("%s\n", encoder_input_paths_split[0].c_str());
  FILE* fp = fopen(encoder_input_paths_split[0].c_str(), "rb");
  if (fp == NULL) {
    perror("open failed!");
    return -1;
  }
  fread(input_data, input_attrs_encoder[0].size, 1, fp);
  fclose(fp);
  if (!input_data) {
    return -1;
  }

  rknn_input inputs_encoder[io_num_encoder.n_input];
  memset(inputs_encoder, 0, io_num_encoder.n_input * sizeof(rknn_input));
  inputs_encoder[0].index        = 0;
  inputs_encoder[0].pass_through = 0;
  inputs_encoder[0].type         = (rknn_tensor_type)RKNN_TENSOR_UINT8;
  inputs_encoder[0].fmt          = (rknn_tensor_format)RKNN_TENSOR_NHWC;
  inputs_encoder[0].buf          = input_data;
  inputs_encoder[0].size         = input_attrs_encoder[0].size;

  // Set input
  ret = rknn_inputs_set(ctx_encoder, io_num_encoder.n_input, inputs_encoder);
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return -1;
  }

  rknn_tensor_mem* output_mems_encoder[io_num_encoder.n_output];
  for (uint32_t i = 0; i < io_num_encoder.n_output; ++i) {
    int output_size = output_attrs_encoder[i].n_elems * 2;
    output_mems_encoder[i]  = rknn_create_mem(ctx_encoder, output_size);
  }

// Set output tensor memory
for (uint32_t i = 0; i < io_num_encoder.n_output; ++i) {
  output_attrs_encoder[i].type = RKNN_TENSOR_FLOAT16;
  output_attrs_encoder[i].fmt = RKNN_TENSOR_NHWC;
  ret = rknn_set_io_mem(ctx_encoder, output_mems_encoder[i], &output_attrs_encoder[i]);
  if (ret < 0) {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }
}

// // =============init preprosess========================
//   unsigned char* input_data_preprosess[io_num_preprosess.n_input];
//   int            input_type_preprosess[io_num_preprosess.n_input];
//   int            input_layout_preprosess[io_num_preprosess.n_input];
//   int            input_size_preprosess[io_num_preprosess.n_input];

//   input_data_preprosess[0]   = NULL;
//   input_type_preprosess[0]   = RKNN_TENSOR_FLOAT16;
//   input_layout_preprosess[0] = RKNN_TENSOR_NHWC;
//   input_size_preprosess[0]   = input_attrs_preprosess[0].size;

//   input_data_preprosess[1]   = NULL;
//   input_type_preprosess[1]   = RKNN_TENSOR_FLOAT16;
//   input_layout_preprosess[1] = RKNN_TENSOR_NHWC;
//   input_size_preprosess[1]   = input_attrs_preprosess[1].size;

//   // Load input
//   if (io_num_preprosess.n_input != preprosess_input_paths_split.size()) {
//     printf("input number mismatch! model input num: %d, input paths num: %zu\n", io_num_preprosess.n_input, preprosess_input_paths_split.size());
//     return -1;
//   }
//   for (int i = 0; i < io_num_preprosess.n_input; i++) {
//     input_data_preprosess[i] = new unsigned char[input_attrs_preprosess[i].size]; 
//     printf("%s\n", preprosess_input_paths_split[i].c_str());
//     FILE* fp1 = fopen(preprosess_input_paths_split[i].c_str(), "rb");
//     if (fp1 == NULL) {
//       perror("open failed!");
//       return -1;
//     }
//     fread(input_data_preprosess[i], input_attrs_preprosess[i].size, 1, fp1);
//     fclose(fp1);
//     if (!input_data_preprosess[i]) {
//       return -1;
//     }
//   }

//   rknn_input inputs_preprosess[io_num_preprosess.n_input];
//   memset(inputs_preprosess, 0, io_num_preprosess.n_input * sizeof(rknn_input));
//   for (int i = 0; i < io_num_preprosess.n_input; i++) {
//     inputs_preprosess[i].index        = i;
//     inputs_preprosess[i].pass_through = 0; // Pass through mode
//     inputs_preprosess[i].type         = (rknn_tensor_type)input_type_preprosess[i];
//     inputs_preprosess[i].fmt          = (rknn_tensor_format)input_layout_preprosess[i];
//     inputs_preprosess[i].buf          = input_data_preprosess[i];
//     inputs_preprosess[i].size         = input_size_preprosess[i];
//   }

//   // Set input
//   ret = rknn_inputs_set(ctx_preprosess, io_num_preprosess.n_input, inputs_preprosess);
//   if (ret < 0) {
//     printf("rknn_input_set fail! ret=%d\n", ret);
//     return -1;
//   }
//   int bs_array_preprosess[2] = {2, 2}; // Batch size array for preprosess
//   // Allocate output memory
//   rknn_tensor_mem* output_mems_preprosess[io_num_preprosess.n_output];
//   for (uint32_t i = 0; i < io_num_preprosess.n_output; ++i) {
//     int output_size = output_attrs_preprosess[i].n_elems * bs_array_preprosess[i];
//     output_mems_preprosess[i]  = rknn_create_mem(ctx_preprosess, output_size);
//   }

//   //test
//   output_attrs_preprosess[0].type = RKNN_TENSOR_FLOAT16;
//   output_attrs_preprosess[0].fmt = RKNN_TENSOR_NHWC;

//   output_attrs_preprosess[1].type = RKNN_TENSOR_FLOAT16;
//   output_attrs_preprosess[1].fmt = RKNN_TENSOR_NHWC;

//   // Set output tensor memory
//   for (uint32_t i = 0; i < io_num_preprosess.n_output; ++i) {
//     // set output memory and attribute
//     ret = rknn_set_io_mem(ctx_preprosess, output_mems_preprosess[i], &output_attrs_preprosess[i]);
//     if (ret < 0) {
//       printf("preprosess rknn_set_io_mem fail! ret=%d\n", ret);
//       return -1;
//     }
//   }

// =============init grid sample========================
  std::string flat_idx_path = "/home/jz/catkin_ws/src/rknn_cpp/rknn_simplebev/npy/flat_idx_240_320.bin";

  int            input_type_grid_sample[io_num_grid_sample.n_input];
  int            input_layout_grid_sample[io_num_grid_sample.n_input];

  input_type_grid_sample[0]   = RKNN_TENSOR_FLOAT16;
  input_layout_grid_sample[0] = RKNN_TENSOR_NHWC;

  input_type_grid_sample[1]   = RKNN_TENSOR_FLOAT16;
  input_layout_grid_sample[1] = RKNN_TENSOR_NHWC;

  for (int i = 0; i < io_num_grid_sample.n_input; i++) {
    input_attrs_grid_sample[i].type = (rknn_tensor_type)input_type_grid_sample[i];
    input_attrs_grid_sample[i].fmt = (rknn_tensor_format)input_layout_grid_sample[i];
  }

  unsigned char* flat_idx;
  // Load input
  flat_idx = new unsigned char[input_attrs_grid_sample[1].size]; 
  printf("%s\n", flat_idx_path.c_str());
  FILE* fp_flat_idx = fopen(flat_idx_path.c_str(), "rb");
  if (fp_flat_idx == NULL) {
    perror("open failed!");
    return -1;
  }
  fread(flat_idx, input_attrs_grid_sample[1].size, 1, fp_flat_idx);
  fclose(fp_flat_idx);
  if (!flat_idx) {
    return -1;
  }
  rknn_tensor_mem* flat_idx_mems;
  flat_idx_mems = rknn_create_mem(ctx_grid_sample, input_attrs_grid_sample[1].size_with_stride);
  // Copy input data to input tensor memory
  int width  = input_attrs_grid_sample[1].dims[2];
  int stride = input_attrs_grid_sample[1].w_stride;
  if (width == stride) {
    memcpy(flat_idx_mems->virt_addr, flat_idx, input_attrs_grid_sample[1].size);
  } else { 
    int height  = input_attrs_grid_sample[1].dims[1];
    int channel = input_attrs_grid_sample[1].dims[3];
    // copy from src to dst with stride
    uint8_t* src_ptr = flat_idx;
    uint8_t* dst_ptr = (uint8_t*)flat_idx_mems->virt_addr;
    // width-channel elements
    int src_wc_elems = width * channel;
    int dst_wc_elems = stride * channel;
    for (int h = 0; h < height; ++h) {
      memcpy(dst_ptr, src_ptr, src_wc_elems);
      src_ptr += src_wc_elems;
      dst_ptr += dst_wc_elems;
    }
  }

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx_grid_sample, output_mems_encoder[0], &input_attrs_grid_sample[0]);
  if (ret < 0) {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }
  ret = rknn_set_io_mem(ctx_grid_sample, flat_idx_mems, &input_attrs_grid_sample[1]);

  int bs_array_grid_sample[2] = {2, 2};
  // Allocate output memory
  rknn_tensor_mem* output_mems_grid_sample[io_num_grid_sample.n_output];
  for (uint32_t i = 0; i < io_num_grid_sample.n_output; ++i) {
    int output_size = output_attrs_grid_sample[i].n_elems * bs_array_grid_sample[i];
    output_mems_grid_sample[i]  = rknn_create_mem(ctx_grid_sample, output_size);
  }

  output_attrs_grid_sample[0].type = RKNN_TENSOR_FLOAT16;
  output_attrs_grid_sample[0].fmt = RKNN_TENSOR_NHWC;  
  
  output_attrs_grid_sample[1].type = RKNN_TENSOR_FLOAT16;
  output_attrs_grid_sample[1].fmt = RKNN_TENSOR_UNDEFINED;  

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num_grid_sample.n_output; ++i) {
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx_grid_sample, output_mems_grid_sample[i], &output_attrs_grid_sample[i]);
    if (ret < 0) {
      printf("grid_sample rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }

  // =============init decoder========================
  input_attrs_decoder[0].type = (rknn_tensor_type)RKNN_TENSOR_FLOAT16;
  input_attrs_decoder[0].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
  // input_attrs_decoder[1].type = (rknn_tensor_type)RKNN_TENSOR_FLOAT16;
  // input_attrs_decoder[1].fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx_decoder, output_mems_grid_sample[0], &input_attrs_decoder[0]);
  if (ret < 0) {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }
  // ret = rknn_set_io_mem(ctx_decoder, output_mems_preprosess[0], &input_attrs_decoder[1]);
  // if (ret < 0) {
  //   printf("rknn_set_io_mem fail! ret=%d\n", ret);
  //   return -1;
  // }

  // Allocate output memory
  rknn_tensor_mem* output_mems_decoder[io_num_decoder.n_output];
  for (uint32_t i = 0; i < io_num_decoder.n_output; ++i) {
    int output_size = output_attrs_decoder[i].n_elems * 2;
    output_mems_decoder[i]  = rknn_create_mem(ctx_decoder, output_size);
  }

  output_attrs_decoder[0].type = RKNN_TENSOR_FLOAT16;
  output_attrs_decoder[0].fmt = RKNN_TENSOR_NCHW;  
  
  output_attrs_decoder[1].type = RKNN_TENSOR_FLOAT16;
  output_attrs_decoder[1].fmt = RKNN_TENSOR_NCHW;  

  output_attrs_decoder[2].type = RKNN_TENSOR_FLOAT16;
  output_attrs_decoder[2].fmt = RKNN_TENSOR_NCHW;  

  output_attrs_decoder[3].type = RKNN_TENSOR_FLOAT16;
  output_attrs_decoder[3].fmt = RKNN_TENSOR_NCHW;  

  output_attrs_decoder[4].type = RKNN_TENSOR_FLOAT16;
  output_attrs_decoder[4].fmt = RKNN_TENSOR_NCHW;  

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num_decoder.n_output; ++i) {
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx_decoder, output_mems_decoder[i], &output_attrs_decoder[i]);
    if (ret < 0) {
      printf("decoder rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }


  
  // ret = rknn_run(ctx_preprosess, NULL);
  // if (ret < 0) {
  //   printf("grid_sample rknn_run fail! ret=%d\n", ret);
  //   return -1;
  // }

  auto start = std::chrono::high_resolution_clock::now();

  ret = rknn_run(ctx_encoder, NULL);
  if (ret < 0) {
    printf("grid_sample rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  for(int i=0; i<io_num_encoder.n_output; i++){
        
    printf("encoder output[%d] shape: [", i);
    for(int j=0; j<output_attrs_encoder[i].n_dims; j++){
        printf("%d ", output_attrs_encoder[i].dims[j]);
    }
    printf("]\n");
    
    auto* out_data = (rknpu2::float16 *)output_mems_encoder[i]->virt_addr;
    for(int k=0; k<100; k++){ // 示例只打印前10个数据
        printf("%f ", (float)out_data[k]);
    }
    printf("\n...\n");
}

  // for(int i=0; i<io_num_preprosess.n_output; i++){
  //   printf("preprosess output[%d] data: ", i);
  //   auto* out_data = (rknpu2::float16 *)output_mems_preprosess[i]->virt_addr;
  //   for(int k=0; k<100; k++){ // 示例只打印前10个数据
  //       printf("%f ", (float)out_data[k]);
  //   }
  //   printf("\n...\n");
  // }

  ret = rknn_run(ctx_grid_sample, NULL);
  if (ret < 0) {
    printf("grid_sample rknn_run fail! ret=%d\n", ret);
    return -1;
  }
  ret = rknn_run(ctx_decoder, NULL);
  if (ret < 0) {
    printf("decoder rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Time taken for rknn_run: %lld ms\n", duration.count());

  for(int i=0; i<io_num_grid_sample.n_output; i++){      
    printf("output[%d] shape: [", i);
    for(int j=0; j<output_attrs_grid_sample[i].n_dims; j++){
        printf("%d ", output_attrs_grid_sample[i].dims[j]);
    }
    printf("]\n");
  }

  auto* out_data1 = (int8_t *)output_mems_grid_sample[0]->virt_addr;
  for(int k=0; k<10; k++){ // 示例只打印前10个数据
      printf("%d ", out_data1[k]);
  }
  printf("\n...\n");

  auto* out_data2 = (rknpu2::float16 *)output_mems_grid_sample[1]->virt_addr;
  for(int k=0; k<10; k++){ // 示例只打印前10个数据
      printf("%f ", (float)out_data2[k]);
  }
  printf("\n...\n");

  for(int i=0; i<io_num_decoder.n_output; i++){
    printf("decoder output[%d] data: ", i);
    auto* out_data_decoder = (rknpu2::float16 *)output_mems_decoder[i]->virt_addr;
    for(int k=0; k<100; k++){ // 示例只打印前10个数据
        printf("%f ", (float)out_data_decoder[k]);
    }
    printf("\n...\n");
  }

  // release outputs
  // ret = rknn_outputs_release(ctx_encoder, io_num_encoder.n_output, outputs_encoder);
  for (uint32_t i = 0; i < io_num_encoder.n_output; ++i) {
    rknn_destroy_mem(ctx_encoder, output_mems_encoder[i]);
  }
  // for (uint32_t i = 0; i < io_num_preprosess.n_output; ++i) {
  //   rknn_destroy_mem(ctx_preprosess, output_mems_preprosess[i]);
  // }
  for (uint32_t i = 0; i < io_num_grid_sample.n_output; ++i) {
    rknn_destroy_mem(ctx_grid_sample, output_mems_grid_sample[i]);
  }
  for (uint32_t i = 0; i < io_num_decoder.n_output; ++i) {
    rknn_destroy_mem(ctx_decoder, output_mems_decoder[i]);
  }

  // ret = rknn_outputs_release(ctx_grid_sample, io_num_grid_sample.n_output, outputs_grid_sample);

  // destroy
  rknn_destroy(ctx_encoder);
  rknn_destroy(ctx_grid_sample);
  // rknn_destroy(ctx_preprosess);
  rknn_destroy(ctx_decoder);
  // for (int i = 0; i < io_num_encoder.n_input; i++) {
  //   free(input_data[i]);
  // } #TODO
  // for (int i = 0; i < io_num_preprosess.n_input; i++) {
  //   free(input_data_preprosess[i]);
  // }

  return 0;
}
