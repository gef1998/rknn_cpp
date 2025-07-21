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
  int ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, 
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
  char* merge_path = argv[1];
  char* merge_input_paths = argv[2];

  std::vector<std::string> merge_input_paths_split = split(merge_input_paths, "#");

  rknn_context ctx_merge;
  // Get sdk and driver version

  if (init_rknn_model(&ctx_merge, merge_path, "ctx_merge") < 0){ return -1; }

  rknn_sdk_version sdk_ver;
  int ret = rknn_query(ctx_merge, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

  rknn_input_output_num io_num_merge;
  if (query_model_io_num(ctx_merge, io_num_merge, "merge") < 0) {return -1;}

  rknn_tensor_attr input_attrs_merge[io_num_merge.n_input];

  if (query_input_attributes(ctx_merge, &io_num_merge, "merge", input_attrs_merge) < 0) {return -1;}

  // 声明输出属性数组
  rknn_tensor_attr output_attrs_merge[io_num_merge.n_output];

// 统一查询输出属性
  if (query_output_attributes(ctx_merge, &io_num_merge, "merge", output_attrs_merge) < 0) {return -1;}

  rknn_custom_string merge_str;
  if (query_custom_string(ctx_merge, "merge", &merge_str) < 0) {return -1;}
  

  // =============init merge========================
  unsigned char* input_data[io_num_merge.n_input];
  int            input_type[io_num_merge.n_input];
  int            input_layout[io_num_merge.n_input];
  int            input_size[io_num_merge.n_input];

  input_data[0]   = NULL;
  input_type[0]   = RKNN_TENSOR_UINT8;
  input_layout[0] = RKNN_TENSOR_NHWC;
  input_size[0]   = input_attrs_merge[0].size;
  
  // Load input
  if (io_num_merge.n_input != merge_input_paths_split.size()) {
    printf("input number mismatch! model input num: %d, input paths num: %zu\n", io_num_merge.n_input, merge_input_paths_split.size());
    return -1;
  }

  // ===============通用API================
  for (int i = 0; i < 1; i++) {
    input_data[i] = new unsigned char[input_attrs_merge[i].size];

    printf("%s\n", merge_input_paths_split[i].c_str());

    FILE* fp = fopen(merge_input_paths_split[i].c_str(), "rb");
    if (fp == NULL) {
      perror("open failed!");
      return -1;
    }

    fread(input_data[i], input_attrs_merge[i].size, 1, fp);
    fclose(fp);
    if (!input_data[i]) {
      return -1;
    }
  }

  rknn_input inputs_merge[io_num_merge.n_input];
  memset(inputs_merge, 0, io_num_merge.n_input * sizeof(rknn_input));

  inputs_merge[0].index        = 0;
  inputs_merge[0].pass_through = 0;
  inputs_merge[0].type         = (rknn_tensor_type)input_type[0];
  inputs_merge[0].fmt          = (rknn_tensor_format)input_layout[0];
  inputs_merge[0].buf          = input_data[0];
  inputs_merge[0].size         = input_size[0];


  // Set input
  ret = rknn_inputs_set(ctx_merge, io_num_merge.n_input, inputs_merge);
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return -1;
  }

  // Allocate output memory
  rknn_tensor_mem* output_mems_merge[io_num_merge.n_output];
  for (uint32_t i = 0; i < io_num_merge.n_output; ++i) {
    int output_size = output_attrs_merge[i].n_elems * 2;
    output_mems_merge[i]  = rknn_create_mem(ctx_merge, output_size);
  }

  output_attrs_merge[0].type = RKNN_TENSOR_FLOAT16;
  output_attrs_merge[0].fmt = RKNN_TENSOR_NCHW;  
  
  output_attrs_merge[1].type = RKNN_TENSOR_FLOAT16;
  output_attrs_merge[1].fmt = RKNN_TENSOR_NCHW;  

  output_attrs_merge[2].type = RKNN_TENSOR_FLOAT16;
  output_attrs_merge[2].fmt = RKNN_TENSOR_NCHW;  

  output_attrs_merge[3].type = RKNN_TENSOR_FLOAT16;
  output_attrs_merge[3].fmt = RKNN_TENSOR_NCHW;  

  output_attrs_merge[4].type = RKNN_TENSOR_FLOAT16;
  output_attrs_merge[4].fmt = RKNN_TENSOR_NCHW;  

  output_attrs_merge[5].type = RKNN_TENSOR_FLOAT16;
  output_attrs_merge[5].fmt = RKNN_TENSOR_UNDEFINED;  

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num_merge.n_output; ++i) {
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx_merge, output_mems_merge[i], &output_attrs_merge[i]);
    if (ret < 0) {
      printf("merge rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();

  ret = rknn_run(ctx_merge, NULL);
  if (ret < 0) {
    printf("merge rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Time taken for rknn_run: %lld ms\n", duration.count());


  for(int i=0; i<io_num_merge.n_output; i++){
    printf("merge output[%d] data: ", i);
    auto* out_data_merge = (rknpu2::float16 *)output_mems_merge[i]->virt_addr;
    for(int k=0; k<100; k++){ // 示例只打印前10个数据
        printf("%f ", (float)out_data_merge[k]);
    }
    printf("\n...\n");
  }

  // release outputs
  // ret = rknn_outputs_release(ctx_encoder, io_num_encoder.n_output, outputs_encoder);
  for (uint32_t i = 0; i < io_num_merge.n_output; ++i) {
    rknn_destroy_mem(ctx_merge, output_mems_merge[i]);
  }
  // ret = rknn_outputs_release(ctx_grid_sample, io_num_grid_sample.n_output, outputs_grid_sample);

  // destroy
  rknn_destroy(ctx_merge);
  
  return 0;
}
