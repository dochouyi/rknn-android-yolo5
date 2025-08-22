package com.herohan.rknn_yolov5.detection;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.herohan.rknn_yolov5.BlockDetect;
import com.herohan.rknn_yolov5.utils.AssetHelper;
import com.herohan.rknn_yolov5.utils.DefaultModel;
import org.opencv.core.*;

import org.opencv.imgproc.Imgproc;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import com.herohan.rknn_yolov5.detection.DLUtils;

public class BlockDetection {

    public BlockDetect model = new BlockDetect();
    private Mat srcMat; // 用于存储 tif 图片的 Mat

    public BlockDetection(Context context) throws IOException {

        model.init(AssetHelper.assetFilePath(context, DefaultModel.YOLOV5S_FP.name));
    }

    private Mat preprocess(Mat src, int rate) {
        int h = src.rows();
        int w = src.cols();
        int newH = ((h + rate - 1) / rate) * rate;
        int newW = ((w + rate - 1) / rate) * rate;
        Mat resized = new Mat();
        Imgproc.resize(src, resized, new Size(newW, newH));
        resized.convertTo(resized, CvType.CV_32FC3, 3.2 / 255.0, -1.6);
        return resized;
    }

    // 推理函数（伪代码，需用实际模型推理替换）
    private Mat predictPatch2(Mat patch) {
        // 1. 预处理
        Mat input = preprocess(patch, 32);

        // 检查类型并转换
        if (input.type() != CvType.CV_8UC1 && input.type() != CvType.CV_8UC3 && input.type() != CvType.CV_8UC4) {
            Mat temp = new Mat();
            input.convertTo(temp, CvType.CV_8UC3, 255.0); // 如果你需要3通道
            input = temp;
        }

        Bitmap bitmap_input = DLUtils.matToBitmap(input);

        // 2. 模型推理（这里需要你用实际模型推理代码替换）
        int[] result=model.detect(bitmap_input);

        int width = 512;
        int height = 512;

        // 设置为单通道8位
        Mat binary_segmentation = new Mat(height, width, CvType.CV_8UC1);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int value = result[i * width + j];
                byte pixel = (byte) (value == 0 ? 255 : 0); // 0为255，其他为0
                binary_segmentation.put(i, j, new byte[]{pixel});
            }
        }
//        // 打印前10行前10列像素值为例，防止输出太多
//        for (int i = 0; i < Math.min(10, binary_segmentation.rows()); i++) {
//            StringBuilder sb = new StringBuilder();
//            for (int j = 0; j < Math.min(10, binary_segmentation.cols()); j++) {
//                double[] pixel = binary_segmentation.get(i, j);
//                sb.append((int)pixel[0]).append(" ");
//            }
//            Log.d("BinarySegmentation", "Row " + i + ": " + sb.toString());
//        }

        return binary_segmentation;
    }

    private Mat predictPatch(Mat patch) {
        int h = patch.rows();
        int w = patch.cols();
        Mat resultMask = Mat.zeros(h, w, CvType.CV_8UC1);

        // 计算正方形左上角和右下角坐标
        int squareSize = 100;
        int left = (w - squareSize) / 2;
        int top = (h - squareSize) / 2;
        int right = left + squareSize;
        int bottom = top + squareSize;

        // 在resultMask中心绘制正方形
        Imgproc.rectangle(resultMask, new Point(left, top), new Point(right, bottom), new Scalar(255), -1);

        return resultMask;
    }


    // 大图分块推理
    public Mat predict(int patchSize, int overlap) {
        int h = srcMat.rows();
        int w = srcMat.cols();
        Mat resultMask = Mat.zeros(h, w, CvType.CV_8UC1);

        for (int y = 0; y < h; y += patchSize - overlap) {
            for (int x = 0; x < w; x += patchSize - overlap) {
                int yEnd = Math.min(y + patchSize, h);
                int xEnd = Math.min(x + patchSize, w);
                Rect roi = new Rect(x, y, xEnd - x, yEnd - y);
                Mat patch = new Mat(srcMat, roi);

                Mat patchMask = predictPatch(patch);
                Imgproc.resize(patchMask, patchMask, new Size(xEnd - x, yEnd - y), 0, 0, Imgproc.INTER_NEAREST);

                patchMask.copyTo(resultMask.submat(roi));
                Log.d("", "y " + y + "   x " + x);
            }
        }
        return resultMask;
    }


    // 获取有效轮廓
    public List<MatOfPoint> getValidContoursFromBinaryImage(Mat binary, double areaThreshold) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.threshold(binary, binary, 127, 255, Imgproc.THRESH_BINARY);
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<MatOfPoint> validContours = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area >= areaThreshold) {
                validContours.add(contour);
            }
        }
        return validContours;
    }


    // 轮廓转多边形
    public List<List<Point>> getPolygonFromContoursList(List<MatOfPoint> contours, int maxPoints, double epsilonRatio) {
        List<List<Point>> polygons = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double arcLen = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            double epsilon = epsilonRatio * arcLen;
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approx, epsilon, true);
            // 若点数太多，增大epsilon
            while (approx.total() > maxPoints) {
                epsilon *= 1.2;
                Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approx, epsilon, true);
            }
            List<Point> pts = new ArrayList<>();
            for (Point p : approx.toArray()) {
                pts.add(p);
            }
            polygons.add(pts);
        }
        return polygons;
    }

    // 一步获取多边形
    public List<List<Point>> predictAndGetPolygon(double areaThreshold, int maxPoints) {
        Mat binary = predict(512, 8);
        List<MatOfPoint> contours = getValidContoursFromBinaryImage(binary, areaThreshold);
        return getPolygonFromContoursList(contours, maxPoints, 0.01);
    }


    public List<List<Point>> dl_predict(String tiffPath, double areaThreshold, int maxPoints) throws IOException {
        // 读取图片
        srcMat = DLUtils.loadTiffToMat(tiffPath); // 赋值成员属性

        List<List<Point>> resultPolygons = this.predictAndGetPolygon(
                areaThreshold,
                maxPoints
        );

        return resultPolygons;
    }


    public Bitmap draw_polygon(List<List<Point>> polygons) {
        Mat drawMat = srcMat.clone();

        // 通道数处理
        if (drawMat.channels() == 1) {
            Imgproc.cvtColor(drawMat, drawMat, Imgproc.COLOR_GRAY2BGR);
        } else if (drawMat.channels() == 4) {
            // 如果是BGRA
            Imgproc.cvtColor(drawMat, drawMat, Imgproc.COLOR_BGRA2BGR);
        }

        // 填充多边形
        for (List<Point> polygon : polygons) {
            MatOfPoint matOfPoint = new MatOfPoint();
            matOfPoint.fromList(polygon);
            List<MatOfPoint> contourList = new ArrayList<>();
            contourList.add(matOfPoint);
            Imgproc.fillPoly(drawMat, contourList, new Scalar(0, 255, 0)); // 绿色
        }

        return DLUtils.matToBitmap(drawMat);
    }


}
