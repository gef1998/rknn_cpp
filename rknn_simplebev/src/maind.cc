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

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
#include "fp16/Float16.h"

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

static int rknn_GetTopN(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;
  uint32_t top_count = outputCount > topNum ? topNum : outputCount;

  for (i = 0; i < topNum; ++i) {
    pfMaxProb[i] = -FLT_MAX;
    pMaxClass[i] = -1;
  }

  for (j = 0; j < top_count; j++) {
    for (i = 0; i < outputCount; i++) {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4))) {
        continue;
      }

      if (pfProb[i] > *(pfMaxProb + j)) {
        *(pfMaxProb + j) = pfProb[i];
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_image(const char* image_path, rknn_tensor_attr* input_attr)
{
  int req_height  = 0;
  int req_width   = 0;
  int req_channel = 0;

  switch (input_attr->fmt) {
  case RKNN_TENSOR_NHWC:
    req_height  = input_attr->dims[1];
    req_width   = input_attr->dims[2];
    req_channel = input_attr->dims[3];
    break;
  case RKNN_TENSOR_NCHW:
    req_height  = input_attr->dims[2];
    req_width   = input_attr->dims[3];
    req_channel = input_attr->dims[1];
    break;
  default:
    printf("meet unsupported layout\n");
    return NULL;
  }

  int height  = 0;
  int width   = 0;
  int channel = 0;

  unsigned char* image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
  if (image_data == NULL) {
    printf("load image failed!\n");
    return NULL;
  }

  if (width != req_width || height != req_height) {
    unsigned char* image_resized = (unsigned char*)STBI_MALLOC(req_width * req_height * req_channel);
    if (!image_resized) {
      printf("malloc image failed!\n");
      STBI_FREE(image_data);
      return NULL;
    }
    if (stbir_resize_uint8(image_data, width, height, 0, image_resized, req_width, req_height, 0, channel) != 1) {
      printf("resize image failed!\n");
      STBI_FREE(image_data);
      return NULL;
    }
    STBI_FREE(image_data);
    image_data = image_resized;
  }

  return image_data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int            model_len = ftell(fp);
  unsigned char* model     = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp) {
    fclose(fp);
  }
  return model;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage:%s model_path input_path [loop_count]\n", argv[0]);
    return -1;
  }

  char* model_path = argv[1];
  char* input_path = argv[2];

  int loop_count = 1;
  if (argc > 3) {
    loop_count = atoi(argv[3]);
  }

  rknn_context ctx = 0;

  // Load RKNN Model
  int            model_len = 0;
  // unsigned char* model     = load_model(model_path, &model_len);
  // int            ret       = rknn_init(&ctx, model, model_len, 0, NULL);

  int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // Get sdk and driver version
  rknn_sdk_version sdk_ver;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }

  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&input_attrs[i]);
  }

  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&output_attrs[i]);
  }

  // Get custom string
  rknn_custom_string custom_string;
  ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("custom string: %s\n", custom_string.string);

  unsigned char*     input_data   = NULL;
  rknn_tensor_type   input_type   = RKNN_TENSOR_UINT8;
  rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;

  // // Load image
  // input_data = load_image(input_path, &input_attrs[0]);
    input_data = new unsigned char[input_attrs[0].size];
    printf("/home/jz/catkin_ws/src/rknn_cpp/rknn_simplebev/npy/rgb_camXs.bin");
    FILE* fp = fopen("/home/jz/catkin_ws/src/rknn_cpp/rknn_simplebev/npy/rgb_camXs.bin", "rb");
    if (fp == NULL) {
      perror("open failed!");
      return -1;
    }
    fread(input_data, input_attrs[0].size, 1, fp);
    fclose(fp);
    if (!input_data) {
      return -1;
    }

  // Create input tensor memory
  rknn_tensor_mem* input_mems[1];
  // default input type is int8 (normalize and quantize need compute in outside)
  // if set uint8, will fuse normalize and quantize to npu
  input_attrs[0].type = input_type;
  // default fmt is NHWC, npu only support NHWC in zero copy mode
  input_attrs[0].fmt = input_layout;

  input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

  // Copy input data to input tensor memory
  int width  = input_attrs[0].dims[2];
  int stride = input_attrs[0].w_stride;

  if (width == stride) {
    memcpy(input_mems[0]->virt_addr, input_data, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
  } else {
    int height  = input_attrs[0].dims[1];
    int channel = input_attrs[0].dims[3];
    // copy from src to dst with stride
    uint8_t* src_ptr = input_data;
    uint8_t* dst_ptr = (uint8_t*)input_mems[0]->virt_addr;
    // width-channel elements
    int src_wc_elems = width * channel;
    int dst_wc_elems = stride * channel;
    for (int h = 0; h < height; ++h) {
      memcpy(dst_ptr, src_ptr, src_wc_elems);
      src_ptr += src_wc_elems;
      dst_ptr += dst_wc_elems;
    }
  }

  // Create output tensor memory
  rknn_tensor_mem* output_mems[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this require float32 to compute top5
    // allocate float32 output tensor
    int output_size = output_attrs[i].n_elems * 2;
    output_mems[i]  = rknn_create_mem(ctx, output_size);
  }

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
  if (ret < 0) {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this require float32 to compute top5
    output_attrs[i].type = RKNN_TENSOR_FLOAT16;
    output_attrs[i].fmt = RKNN_TENSOR_NHWC;
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    printf("output_mems[i].size=%d, output_attrs[i].size=%d\n", output_mems[i]->size,
           output_attrs[i].size);
    if (ret < 0) {
      printf("rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }

  // Run
  printf("Begin perf ...\n");
  for (int i = 0; i < loop_count; ++i) {
    int64_t start_us  = getCurrentTimeUs();
    ret               = rknn_run(ctx, NULL);
    int64_t elapse_us = getCurrentTimeUs() - start_us;
    if (ret < 0) {
      printf("rknn run error %d\n", ret);
      return -1;
    }
    printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
  }

  // Get top 5
  uint32_t topNum = 5;
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    uint32_t MaxClass[topNum];
    float    fMaxProb[topNum];
    auto   buffer    = (rknpu2::float16*)output_mems[i]->virt_addr;
    uint32_t sz        = output_attrs[i].n_elems;
    int      top_count = sz > topNum ? topNum : sz;
    for(int k=0; k<10; k++){ // 示例只打印前10个数据
        printf("%f ", (float)buffer[k]);
    }
    printf("%f ", (float)buffer[100]);

  }

  // Destroy rknn memory
  rknn_destroy_mem(ctx, input_mems[0]);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    rknn_destroy_mem(ctx, output_mems[i]);
  }

  // destroy
  rknn_destroy(ctx);

  if (input_data != nullptr) {
    free(input_data);
  }

  // if (model != nullptr) {
  //   free(model);
  // }

  return 0;
}
