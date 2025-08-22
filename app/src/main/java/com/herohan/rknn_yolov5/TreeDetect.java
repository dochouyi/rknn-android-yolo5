package com.herohan.rknn_yolov5;

import android.graphics.Bitmap;

public class TreeDetect {
    static {
        System.loadLibrary("rknn_treedetection");
    }

    public native boolean init(String modelPath);

    public native int[] detect(Bitmap srtBitmap);

    public native boolean release();
}
