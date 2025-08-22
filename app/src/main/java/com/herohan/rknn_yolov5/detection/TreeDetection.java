package com.herohan.rknn_yolov5.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import com.herohan.rknn_yolov5.TreeDetect;
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



public class TreeDetection {

    public TreeDetect model = new TreeDetect();
    private Mat srcMat; // 用于存储 tif 图片的 Mat

    public TreeDetection(Context context) throws IOException {

        model.init(AssetHelper.assetFilePath(context, DefaultModel.YOLOV5S_FP.name));
    }


    // 调整图像尺寸为16的倍数
    public static Mat resizeToNearest8(Mat img) {
        int h = img.rows();
        int w = img.cols();
        int newW = Math.round(w / 16f) * 16;
        int newH = Math.round(h / 16f) * 16;
        Mat resized = new Mat();
        Imgproc.resize(img, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_LANCZOS4);
        return resized;
    }

    // 局部极大值检测（需要你实现或用第三方库）
    public static List<Point> getLocalMaxima(Mat dotMap, int minDistance, double thresholdAbs) {
        // 归一化
        Core.MinMaxLocResult mmr = Core.minMaxLoc(dotMap);
        Mat normMap = new Mat();
        Core.subtract(dotMap, new Scalar(mmr.minVal), normMap);
        Core.divide(normMap, new Scalar(mmr.maxVal - mmr.minVal + 1e-8), normMap);

        // TODO: 实现局部极大值检测（如用滑动窗口）
        // 这里只返回空列表作为占位
        return new ArrayList<>();
    }

    // 预处理
    private Mat preprocess(Mat cv2Img) {
        Mat imgRgb = new Mat();
        Imgproc.cvtColor(cv2Img, imgRgb, Imgproc.COLOR_BGR2RGB);
        Mat imgResized = resizeToNearest8(imgRgb);
        // TODO: 转为Tensor/NDArray，归一化
        return imgResized;
    }



    // 单块预测
    private List<Point> predictPatch(Mat cv2Img) {
        Mat imgTensor = preprocess(cv2Img);

        // TODO: 模型推理，获得estDotMap（float[][]）
        // float[][] estDotMap = model.forward(imgTensor);

        // 伪代码
        Mat estDotMap = new Mat(); // 你要替换成实际推理输出
        List<Point> pointList = getLocalMaxima(estDotMap, 2, 0.1);

        // 乘以8
        List<Point> scaledPoints = new ArrayList<>();
        for (Point pt : pointList) {
            scaledPoints.add(new Point(pt.x * 8, pt.y * 8));
        }
        return scaledPoints;
    }




    // 大图分块预测
    public List<Point> predict(Mat cv2Img, int patchSize) {
        int h = cv2Img.rows();
        int w = cv2Img.cols();
        List<Point> bigPicPointList = new ArrayList<>();
        for (int y = 0; y < h; y += patchSize) {
            for (int x = 0; x < w; x += patchSize) {
                int yEnd = Math.min(y + patchSize, h);
                int xEnd = Math.min(x + patchSize, w);
                Rect roi = new Rect(x, y, xEnd - x, yEnd - y);
                Mat patch = new Mat(cv2Img, roi);

                List<Point> pointList = predictPatch(patch);
                pointList = convertPointList(pointList, y, x);

                for (Point pt : pointList) {
                    int px = (int)pt.x, py = (int)pt.y;
                    if (py >= 0 && py < h && px >= 0 && px < w) {
                        double[] pixel = cv2Img.get(py, px);
                        if (isBlackOrWhite(pixel)) continue;
                        if (cv2Img.channels() == 4 && pixel[3] == 0) continue;
                        bigPicPointList.add(pt);
                    }
                }
            }
        }
        return bigPicPointList;
    }

    // 坐标偏移
    public List<Point> convertPointList(List<Point> pointList, int y, int x) {
        List<Point> out = new ArrayList<>();
        for (Point pt : pointList) {
            out.add(new Point(pt.x + x, pt.y + y));
        }
        return out;
    }

    // 乘以缩放系数
    public List<Point> predictPoints(Mat cv2Img, double times) {
        List<Point> points = predict(cv2Img, 512);
        List<Point> result = new ArrayList<>();
        for (Point pt : points) {
            result.add(new Point((int)(pt.x * times), (int)(pt.y * times)));
        }
        return result;
    }


    // 判断像素是否全黑或全白
    private boolean isBlackOrWhite(double[] pixel) {
        if (pixel == null) return true;
        boolean allZero = true, all255 = true;
        for (int i = 0; i < 3 && i < pixel.length; i++) {
            if (pixel[i] != 0) allZero = false;
            if (pixel[i] != 255) all255 = false;
        }
        return allZero || all255;
    }


    public List<Point> dl_predict(String tiffPath) throws IOException {
        srcMat = DLUtils.loadTiffToMat(tiffPath); // 赋值成员属性

        double zoomTimes = 2.5;
        int height = srcMat.rows();
        int width = srcMat.cols();
        int newW = (int)(width / zoomTimes);
        int newH = (int)(height / zoomTimes);
        Mat resized = new Mat();
        Imgproc.resize(srcMat, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_AREA);

        return predictPoints(resized, zoomTimes);
    }

}
