package com.example.readysetset.algorithm;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class Segmenter {
    public Mat mScaled;
    private int width;
    private int height;

    public Segmenter(int width, int height, double scale) {
        this.width = (int)(width*scale);
        this.height = (int)(height*scale);

        mScaled = new Mat(width, height, CvType.CV_8UC1);

    }

    public void detection_candidates(Mat frame) {
        Log.d("DEBUG", frame.size().toString());
        Log.e("SET", frame.type() + "");
        // TODO: this doesn't work
        Imgproc.resize(frame, mScaled, new Size(width, height));
        //Imgproc.GaussianBlur(mScaled, mScaled);
    }
}
