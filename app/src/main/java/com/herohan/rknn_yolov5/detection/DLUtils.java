package com.herohan.rknn_yolov5.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class DLUtils {
    public static String copyAssetToCache(Context context, String assetName) throws IOException {
        InputStream inputStream = context.getAssets().open(assetName);
        File outFile = new File(context.getCacheDir(), assetName);
        FileOutputStream outputStream = new FileOutputStream(outFile);

        byte[] buffer = new byte[4096];
        int length;
        while ((length = inputStream.read(buffer)) > 0) {
            outputStream.write(buffer, 0, length);
        }
        outputStream.close();
        inputStream.close();

        return outFile.getAbsolutePath();
    }

    public static Mat loadTiffToMat(String tiffPath) {
        Mat mat = Imgcodecs.imread(tiffPath, Imgcodecs.IMREAD_UNCHANGED); // 或 IMREAD_COLOR
        if (mat.empty()) {
            Log.e("TIFF", "OpenCV imread failed!");
        }
        return mat;
    }

    public static Bitmap matToBitmap(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        return bitmap;
    }
}
