#ifndef _RKNN_DEMO_YOLOV5_H_
#define _RKNN_DEMO_YOLOV5_H_

#include "utils/common.h"




int init_model(const char* model_path, rknn_app_context_t* app_ctx);

int release_model(rknn_app_context_t* app_ctx);

int inference_model(rknn_app_context_t* app_ctx, image_buffer_t* img, int* output_buffer);

#endif //_RKNN_DEMO_YOLOV5_H_