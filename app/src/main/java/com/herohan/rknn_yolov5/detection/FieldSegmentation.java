package com.herohan.rknn_yolov5.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.herohan.rknn_yolov5.FieldSegment;
import com.herohan.rknn_yolov5.utils.AssetHelper;
import com.herohan.rknn_yolov5.utils.DefaultModel;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.*;



public class FieldSegmentation {
    private static final int[][] COLOR_MAP = {
            {0, 0, 128},     // background
            {144, 238, 144}, // crop
            {0, 100, 0},     // tree
            {192, 192, 192}, // road
            {255, 0, 0},     // pole
            {255, 255, 0},   // building
            {128, 0, 128}    // water
    };

    private FieldSegment model = new FieldSegment();
    private Mat srcMat; // 用于存储 tif 图片的 Mat

    public FieldSegmentation(Context context) throws IOException {

        model.init(AssetHelper.assetFilePath(context, DefaultModel.YOLOV5S_FP.name));
    }
    // 预处理
    public float[] preprocess(Mat img) {
        Mat resized = new Mat();
        Imgproc.resize(img, resized, new Size(inputWidth, inputHeight));
        resized.convertTo(resized, CvType.CV_32FC3, 1.0 / 255);

        // HWC to CHW
        List<Mat> channels = new ArrayList<>();
        Core.split(resized, channels);
        float[] data = new float[inputHeight * inputWidth * 3];
        for (int c = 0; c < 3; c++) {
            channels.get(c).get(0, 0, data, c * inputHeight * inputWidth, inputHeight * inputWidth);
        }
        return data;
    }


    // 后处理
    public int[][] postprocess(float[] output, int outHeight, int outWidth, int numClasses) {
        int[][] mask = new int[outHeight][outWidth];
        for (int i = 0; i < outHeight; i++) {
            for (int j = 0; j < outWidth; j++) {
                int maxIdx = 0;
                float maxVal = -Float.MAX_VALUE;
                for (int c = 0; c < numClasses; c++) {
                    float val = output[c * outHeight * outWidth + i * outWidth + j];
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = c;
                    }
                }
                mask[i][j] = maxIdx;
            }
        }
        return mask;
    }

    // 彩色化
    public Mat colorize(int[][] mask) {
        int h = mask.length;
        int w = mask[0].length;
        Mat colorImg = new Mat(h, w, CvType.CV_8UC3);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int idx = mask[i][j];
                int[] color = COLOR_MAP[idx];
                double[] bgr = {color[2], color[1], color[0]}; // OpenCV: BGR
                colorImg.put(i, j, bgr);
            }
        }
        return colorImg;
    }


    // 单patch推理
    public int[][] predictPatch(Mat patch) throws Exception {
        float[] inputData = preprocess(patch);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData),
                new long[]{1, 3, inputHeight, inputWidth});
        OrtSession.Result result = session.run(Collections.singletonMap(session.getInputNames().iterator().next(), inputTensor));
        float[][][][] outputArr = (float[][][][]) result.get(0).getValue();


//        model.detect()

        // 假设输出为 (1, num_classes, H, W)
        int numClasses = outputArr[0].length;
        int outH = outputArr[0][0].length;
        int outW = outputArr[0][0][0].length;
        float[] flatOut = new float[numClasses * outH * outW];
        int idx = 0;
        for (int c = 0; c < numClasses; c++)
            for (int i = 0; i < outH; i++)
                for (int j = 0; j < outW; j++)
                    flatOut[idx++] = outputArr[0][c][i][j];
        return postprocess(flatOut, outH, outW, numClasses);
    }

    // 大图分块推理
    public int[][] predictLargeImage(Mat img, int patchSize, int overlap) throws Exception {
        int h = img.rows();
        int w = img.cols();
        int[][] resultMask = new int[h][w];

        for (int y = 0; y < h; y += (patchSize - overlap)) {
            for (int x = 0; x < w; x += (patchSize - overlap)) {
                int yEnd = Math.min(y + patchSize, h);
                int xEnd = Math.min(x + patchSize, w);
                Rect roi = new Rect(x, y, xEnd - x, yEnd - y);
                Mat patch = new Mat(img, roi);
                int[][] patchMask = predictPatch(patch);

                // resize patchMask to (yEnd-y, xEnd-x)
                Mat patchMaskMat = new Mat(patchMask.length, patchMask[0].length, CvType.CV_8UC1);
                for (int i = 0; i < patchMask.length; i++)
                    for (int j = 0; j < patchMask[0].length; j++)
                        patchMaskMat.put(i, j, patchMask[i][j]);
                Mat resizedMask = new Mat();
                Imgproc.resize(patchMaskMat, resizedMask, new Size(xEnd - x, yEnd - y), 0, 0, Imgproc.INTER_NEAREST);

                // 合并结果
                for (int i = y; i < yEnd; i++)
                    for (int j = x; j < xEnd; j++)
                        resultMask[i][j] = (int) resizedMask.get(i - y, j - x)[0];
            }
        }
        return resultMask;
    }


}
