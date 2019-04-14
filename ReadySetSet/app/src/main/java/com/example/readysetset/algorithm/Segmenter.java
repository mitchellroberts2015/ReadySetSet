package com.example.readysetset.algorithm;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;

public class Segmenter {
    // stores the main operations
    private Mat mMat;
    private Mat mOrigSmall;
    private Mat mResults;
    private Mat mBinaryDisplay;
    private Mat mBuffer;
    
    // the original mHeight and mWidth of the input frame
    private int mWidth;
    private int mHeight;
    private ArrayList<MatOfPoint> mContours;
    private MatOfPoint2f mHierarchy;
    private ArrayList<MatOfPoint> mContoursFiltered;
    private ArrayList<List<Point>> mCandidates;
    
    // a list of useful debugging images
    public ArrayList<Mat> mDebugMats;


    private static Mat kernel = Mat.ones(3, 3, CvType.CV_32F);

    public Segmenter(int width, int height, double scale) {
        this.width = (int)(width*scale);
        this.height = (int)(height*scale);

        int width_small = (int)(width*scale);
        int height_small = (int)(height*scale);


        mMat = new Mat(height_small, width_small, CvType.CV_8UC1);
        mBuffer = new Mat(height_small, width_small, CvType.CV_8UC4);
        mOrigSmall = new Mat(height_small, width_small, CvType.CV_8UC4);
        mResults = new Mat(height, width, CvType.CV_8UC4);
        mBinaryDisplay = new Mat(height, width, CvType.CV_8UC4);
        mContours = new ArrayList<>(30);
        mCandidates = new ArrayList<>(30);
        mContoursFiltered = new ArrayList<>(30);
        mHierarchy = new MatOfPoint2f();
        mDebugMats = new ArrayList<>();
        mDebugMats.add(mResults);
        mDebugMats.add(mBinaryDisplay);
    }

    public void detection_candidates(Mat frame) {
        // make binary image
        Imgproc.resize(frame, mOrigSmall, new Size(mWidth /3, mHeight /3));
        Imgproc.cvtColor(mOrigSmall, mMat, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.GaussianBlur(mMat, mMat, new Size(5,5), 0);
        Imgproc.threshold(mMat, mMat, 150, 255, Imgproc.THRESH_BINARY);
        Imgproc.morphologyEx(mMat, mMat, Imgproc.MORPH_OPEN, kernel);


        // find contours
        mContours.clear();
        Imgproc.findContours(mMat, mContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // filter contours
        mContoursFiltered.clear();
        mCandidates.clear();
        for (MatOfPoint contour : mContours) {
            MatOfPoint2f buffer = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f outBuffer = new MatOfPoint2f();
            Imgproc.approxPolyDP(buffer, outBuffer, 10, true);

            if (outBuffer.toList().size() != 4) continue;
            double area = Imgproc.contourArea(outBuffer);
            if (area < 1000 || area > 20000) continue;
            double angle = angleScore(outBuffer.toList());
            Log.e("SET", angle + "");
            if (angle > Math.PI / 10) continue;

            MatOfPoint mat = new MatOfPoint(outBuffer.toArray());
            if(!mat.empty()) {
                mContoursFiltered.add(mat);
                mCandidates.add(mat.toList());
            }
        }

        if (mCandidates.size() > 0) {
            Log.e("SET", "trying the thing...");
            Mat card = new Mat(150, 250, CvType.CV_8UC4);
           // card.setTo(new Scalar(0,255,255,127));

            getImage(mOrigSmall, card, mCandidates.get(0));
            Mat submat = mOrigSmall.submat(new Rect(0, 0, 250, 150));
            card.copyTo(submat);
            //for (int i = 0; i < 150; i++) {
            //    for (int j = 0; j < 250; j++) {
                    //mOrigSmall.put(i, j, card.get(i, j));
                //}
            //}
        }

        // save the binary image to mBinaryDisplay
        Imgproc.cvtColor(mMat, mBuffer, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.resize(mBuffer, mBinaryDisplay, new Size(mWidth, mHeight));

        Imgproc.drawContours(mOrigSmall, mContoursFiltered, -1, new Scalar(0,255,0), 3);
        Imgproc.resize(mOrigSmall, mResults, new Size(mWidth, mHeight));
    }

    public void release() {
        mMat.release();
        mOrigSmall.release();
        mResults.release();
        mBinaryDisplay.release();
        mBuffer.release();
        for (MatOfPoint mat : mContours) {
            mat.release();
        }
        for (MatOfPoint mat : mContoursFiltered) {
            mat.release();
        }
        mHierarchy.release();
    }


    HashMap<Point,Double> vals = new HashMap<>(4);
    private void orderPoints(List<Point> pts) {
        Point top_left = null;
        Point bottom_right = null;
        double sum_tl = Double.MAX_VALUE;
        double sum_br = 0;

        for (Point p : pts) {
            double new_sum = p.x + p.y;
            if (new_sum < sum_tl) {
                top_left = p;
                sum_tl = new_sum;
                Log.e("SET", "set top left");
            }
            if (new_sum > sum_br) {
                bottom_right = p;
                sum_br = new_sum;
                Log.e("SET", "set bottom right");
            }
        }

        Point top_right = null;
        Point bottom_left = null;
        double diff_tr = Double.MAX_VALUE;
        double diff_bl = -Double.MAX_VALUE;

        for (Point p : pts) {
            double new_diff = p.x - p.y;
            Log.e("SET", "diff: " + new_diff + ", diff_bl: " + diff_bl);
            if (new_diff < diff_tr) {
                top_right = p;
                diff_tr = new_diff;
                Log.e("SET", "set top right");
            }
            if (new_diff > diff_bl) {
                bottom_left = p;
                diff_bl = new_diff;
                Log.e("SET", "set bottom left");
            }
        }
        double dist1 = Math.pow(top_left.x - top_right.x, 2) + Math.pow(top_left.y - top_right.y, 2);
        double dist2 = Math.pow(top_right.x - bottom_right.x, 2) + Math.pow(top_right.y - bottom_right.y, 2);

        if (dist1 > dist2) {
            pts.set(0, top_left);
            pts.set(1, top_right);
            pts.set(2, bottom_right);
            pts.set(3, bottom_left);
        } else {
            pts.set(0, top_right);
            pts.set(1, bottom_right);
            pts.set(2, bottom_left);
            pts.set(3, top_left);
        }
    }

    MatOfPoint2f dst = new MatOfPoint2f(
            new Point(0, 0),
            new Point(250, 0),
            new Point(250, 150),
            new Point(0, 150)
    );

    private Mat getWarp(List<Point> pts) {
        orderPoints(pts);
        Log.e("SET", "START");
        for (Point p : pts) {
            Log.e("SET", p.toString());
        }
        Log.e("SET", "END");
        MatOfPoint2f src = new MatOfPoint2f(
                pts.get(0),
                pts.get(1),
                pts.get(2),
                pts.get(3)
        );
        return Imgproc.getPerspectiveTransform(src, dst);
    }

    private void getImage(Mat img, Mat dst, List<Point> pts) {
        Mat warp = getWarp(pts);

        Imgproc.warpPerspective(img, dst, warp, dst.size());
    }

    Point v1 = new Point(0, 0);
    Point v2 = new Point(0, 0);

    /**
     * Calculates the angle at each corner,
     * for each computes abs(angle - 90)
     * returns the largest
     * @param points
     * @return
     */
    private double angleScore(List<Point> points) {
        double score = 0;

        for (int j = 0; j < points.size(); j++) {
            int i = (j - 1) % points.size();
            if (i < 0) i += points.size();
            int k = (j + 1) % points.size();

            Point p1 = points.get(i);
            Point p2 = points.get(j);
            Point p3 = points.get(k);

            //Point v1 = new Point(p1.x - p2.x, p1.y - p2.y);
            //Point v2 = new Point(p3.x - p2.x, p3.y - p2.y);
            v1.x = p1.x - p2.x;
            v1.y = p1.y - p2.y;

            v2.x = p3.x - p2.x;
            v2.y = p3.y - p2.y;


            double angle = Math.acos(v1.dot(v2) / Math.sqrt(v1.dot(v1)) / Math.sqrt(v2.dot(v2)));
            double new_score = Math.abs(angle - Math.PI/2);
            score = Math.max(score, new_score);
        }

        return score;
    }
}
