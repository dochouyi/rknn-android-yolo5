package com.herohan.rknn_yolov5;
import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;


import org.opencv.android.OpenCVLoader;
import java.io.IOException;

import com.herohan.rknn_yolov5.detection.DLUtils;
import com.herohan.rknn_yolov5.detection.BlockDetection;
import com.herohan.rknn_yolov5.detection.TreeDetection;
import com.herohan.rknn_yolov5.detection.FieldSegmentation;




public class MainActivity extends Activity {
    private static final String TAG = MainActivity.class.getSimpleName();

    private ImageView imageView;
    BlockDetection blockDetection;
    TreeDetection treeDetection;
    FieldSegmentation fieldSegmentation;

    public void process_blockDetection(){
        try {
            blockDetection = new BlockDetection(this);
            double areaThreshold = 1000.0; // 你需要根据实际情况调整
            int maxPoints = 10; // 你需要根据实际情况调整

            // 调用预测并获取多边形
            java.util.List<java.util.List<org.opencv.core.Point>> polygons =
                    blockDetection.predictGpsJson("odm_orthophoto_1.tif", areaThreshold, maxPoints);

            // 打印多边形点
            for (int i = 0; i < polygons.size(); i++) {
                java.util.List<org.opencv.core.Point> polygon = polygons.get(i);
                StringBuilder sb = new StringBuilder();
                sb.append("Polygon ").append(i).append(": ");
                for (org.opencv.core.Point pt : polygon) {
                    sb.append(String.format("(%.1f, %.1f) ", pt.x, pt.y));
                }
                Log.i(TAG, sb.toString());
            }

            Bitmap resultBitmap = blockDetection.draw_polygon(polygons);
            // 你可以在这里进一步处理，比如可视化多边形等

            imageView.setImageBitmap(resultBitmap);

        } catch (IOException e) {
            Log.e(TAG, "Error processing tiff file", e);
        }
    }
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        imageView = (ImageView) findViewById(R.id.imageView);

        // 1. 初始化 OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed!");
            return;
        }

        this.process_blockDetection();

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        boolean retInit = blockDetection.model.release();
        if (!retInit) {
            Log.e(TAG, "BlockDetect Release failed");
        }
    }
}
