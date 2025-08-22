#include <android/log.h>
#include <android/bitmap.h>

#include <jni.h>

#include <sys/time.h>
#include <string>
#include <vector>
#include "detection.h"

#include "utils/image_drawing.h"

extern "C" {

static rknn_app_context_t rknn_app_ctx;



JNIEXPORT jint
JNI_OnLoad(JavaVM *vm, void *reserved) {

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {

}

JNIEXPORT jboolean JNICALL
Java_com_herohan_rknn_1yolov5_TreeDetect_init(JNIEnv *env, jobject thiz, jstring jmodel_path) {

    const char *modelPath = (env->GetStringUTFChars(jmodel_path, 0));

    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_model(modelPath, &rknn_app_ctx);

    return JNI_TRUE;
}

JNIEXPORT jintArray JNICALL
Java_com_herohan_rknn_1yolov5_TreeDetect_detect(JNIEnv *env, jobject thiz, jobject jbitmap) {
    AndroidBitmapInfo dstInfo;

    if (ANDROID_BITMAP_RESULT_SUCCESS != AndroidBitmap_getInfo(env, jbitmap, &dstInfo)) {
        LOGE("get bitmap info failed");
        return JNI_FALSE;
    }

    void *dstBuf;
    if (ANDROID_BITMAP_RESULT_SUCCESS != AndroidBitmap_lockPixels(env, jbitmap, &dstBuf)) {
        LOGE("lock dst bitmap failed");
        return JNI_FALSE;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    src_image.width = dstInfo.width;
    src_image.height = dstInfo.height;
    src_image.format = IMAGE_FORMAT_RGBA8888;
    src_image.virt_addr = static_cast<unsigned char *>(dstBuf);
    src_image.size = dstInfo.width * dstInfo.height * 4;



    int out_h = 512, out_w = 512;
    int output_size = out_h * out_w;
    int *output_buffer = (int*)malloc(output_size * sizeof(int));
    if (!output_buffer) {
        LOGE("malloc output_buffer failed");
        AndroidBitmap_unlockPixels(env, jbitmap);
        return NULL;
    }

    inference_model(&rknn_app_ctx, &src_image, output_buffer);

    AndroidBitmap_unlockPixels(env, jbitmap);
    // 创建 Java int[] 数组
    jintArray resultArray = env->NewIntArray(output_size);
    if (resultArray == NULL) {
        LOGE("NewIntArray failed");
        free(output_buffer);
        return NULL;
    }

    // 拷贝 output_buffer 到 Java 数组
    env->SetIntArrayRegion(resultArray, 0, output_size, output_buffer);

    // 释放 C 层内存
    free(output_buffer);

    // 返回 Java int[]
    return resultArray;
}


JNIEXPORT jboolean JNICALL
Java_com_herohan_rknn_1yolov5_TreeDetect_release(JNIEnv *env, jobject thiz) {

    release_model(&rknn_app_ctx);

    return true;
}

}