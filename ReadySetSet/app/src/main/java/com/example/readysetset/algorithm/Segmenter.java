package com.example.readysetset.algorithm;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

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
    private ArrayList<MatOfPoint> mCandidates;
    
    // a list of useful debugging images
    public ArrayList<Mat> mDebugMats;


    private static Mat kernel = Mat.ones(3, 3, CvType.CV_32F);

    public Segmenter(int width, int height, double scale) {
        this.mWidth = width;
        this.mHeight = height;

        int width_small = (int)(width*scale);
        int height_small = (int)(height*scale);

        mMat = new Mat(width_small, height_small, CvType.CV_8UC1);
        mBuffer = new Mat(width_small, height_small, CvType.CV_8UC4);
        mOrigSmall = new Mat(width_small, height_small, CvType.CV_8UC4);
        mResults = new Mat(width, height, CvType.CV_8UC4);
        mBinaryDisplay = new Mat(width, height, CvType.CV_8UC4);
        mContours = new ArrayList<>(30);
        mCandidates = new ArrayList<>(30);
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

        // save the binary image to mBinaryDisplay
        Imgproc.cvtColor(mMat, mBuffer, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.resize(mBuffer, mBinaryDisplay, new Size(mWidth, mHeight));

        // find contours
        mContours.clear();
        Imgproc.findContours(mMat, mContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // filter contours
        mCandidates.clear();
        for (MatOfPoint contour : mContours) {
            MatOfPoint2f buffer = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f outBuffer = new MatOfPoint2f();
            Imgproc.approxPolyDP(buffer, outBuffer, 10, true);

            if (outBuffer.toList().size() != 4) continue;
            double area = Imgproc.contourArea(outBuffer);
            if (area < 1000 || area > 20000) continue;

            MatOfPoint mat = new MatOfPoint(outBuffer.toArray());
            if(!mat.empty()) mCandidates.add(mat);
        }

        Imgproc.drawContours(mOrigSmall, mCandidates, -1, new Scalar(0,255,0), 3);
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
        for (MatOfPoint mat : mCandidates) {
            mat.release();
        }
        mHierarchy.release();
    }
}
