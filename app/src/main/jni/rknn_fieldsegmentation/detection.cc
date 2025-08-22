#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "detection.h"
#include "utils/common.h"
#include "utils/file_utils.h"
#include "utils/image_utils.h"



int init_model(const char *model_path, rknn_app_context_t *app_ctx) {

    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    model_len = read_data_from_file(model_path, &model);

    rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));

    }
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));

    }
    app_ctx->rknn_ctx = ctx;
    app_ctx->is_quant = false;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *) malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *) malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    app_ctx->model_height = input_attrs[0].dims[1];
    app_ctx->model_width = input_attrs[0].dims[2];
    app_ctx->model_channel = input_attrs[0].dims[3];


    LOGI("%d", app_ctx->model_height);
    LOGI("%d", app_ctx->model_width);
    LOGI("%d", app_ctx->model_channel);

//    rknn_sdk_version version;
//    rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
//    LOGI("%s", version.api_version);
//    LOGI("%s", version.drv_version);

//    for (int i = 0; i < io_num.n_input; i++) {
//        LOGI("Input %d:", i);
//        LOGI("  name: %s", input_attrs[i].name);
//        LOGI("  n_dims: %d", input_attrs[i].n_dims);
//        LOGI("  dims: %d %d %d %d",
//             input_attrs[i].dims[0],
//             input_attrs[i].dims[1],
//             input_attrs[i].dims[2],
//             input_attrs[i].dims[3]);
//        LOGI("  type: %d", input_attrs[i].type);
//        LOGI("  fmt: %d", input_attrs[i].fmt);
//    }
//
//    for (int i = 0; i < io_num.n_output; i++) {
//        LOGI("Output %d:", i);
//        LOGI("  name: %s", output_attrs[i].name);
//        LOGI("  n_dims: %d", output_attrs[i].n_dims);
//        LOGI("  dims: %d %d %d %d",
//             output_attrs[i].dims[0],
//             output_attrs[i].dims[1],
//             output_attrs[i].dims[2],
//             output_attrs[i].dims[3]);
//        LOGI("  type: %d", output_attrs[i].type);
//        LOGI("  fmt: %d", output_attrs[i].fmt);
//    }

    return 0;
}

int release_model(rknn_app_context_t *app_ctx) {
    if (app_ctx->rknn_ctx != 0) {
        // 9.销毁 RKNN
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    return 0;
}


int inference_model(rknn_app_context_t *app_ctx, image_buffer_t *img,
                    int *output_buffer)
{
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    void *output_data[app_ctx->io_num.n_output];


    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *) malloc(dst_img.size);

    // 3.对输入进行前处理
    convert_image_with_letterbox(img, &dst_img, &letter_box, 114);

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;

    // 4.设置输入数据
    rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);

    // 5.进行模型推理
    rknn_run(app_ctx->rknn_ctx, nullptr);

    // Get Output
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    // 6.获取推理结果数据
    rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        output_data[i] = outputs[i].buf;
    }

    float* data = (float*)output_data[0];

    int out_c = 7;
    int out_h = 512;
    int out_w = 512;

    // 计算 argmax 并存入 output_buffer
    for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
            int max_idx = 0;
            float max_val = data[0 * out_h * out_w + h * out_w + w];
            for (int c = 1; c < out_c; ++c) {
                float val = data[c * out_h * out_w + h * out_w + w];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            output_buffer[h * out_w + w] = max_idx;
        }
    }

    // 8.释放输出数据内存
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    if (dst_img.virt_addr != NULL) {
        free(dst_img.virt_addr);
    }

    return 0;
}

