package com.example.readysetset.algorithm;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class Tracker {
    static public class Card {
        private ArrayList<List<Point>> mHistory;
        private int mSize;
        private int mI;
        private int[] mFeatures;
        private Rect mRect;
        private Point p1, p2, p3, p4;
        private List<Point> mPts;
        private Date mLastUpdate;


        public Card(List<Point> pts, int size) {
            mHistory = new ArrayList<>(size);
            mHistory.add(pts);
            mSize = size;
            mI = 0;
            p1 = new Point(0, 0);
            p2 = new Point(0, 0);
            p3 = new Point(0, 0);
            p4 = new Point(0, 0);
            mPts = new MatOfPoint(p1, p2, p3, p4).toList();
        }

        private double getDist(Point p1, Point p2) {
            return Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2);
        }

        public void detection(List<Point> pts) {
            List<Point> last_pts = mHistory.get(mI);
            Point pos[] = new Point[4];
            for (Point p1 : pts) {
                int best_i = 0;
                double best_dist = Double.MAX_VALUE;
                for (int i = 0; i < last_pts.size(); i++) {
                    double dist = getDist(p1, last_pts.get(i));
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_i = i;
                    }
                }
                pos[best_i] = p1;
            }

            for (int i = 0; i < pos.length; i++) {
                pts.set(i, pos[i]);
            }

            mI = (mI + 1) % mSize;
            mHistory.set(mI, pts);
            mLastUpdate = new Date();
        }

        public List<Point> getPts() {
            p1.x = 0;
            p1.y = 0;
            p2.x = 0;
            p2.y = 0;
            p3.x = 0;
            p3.y = 0;
            p4.x = 0;
            p4.y = 0;
            for (List<Point> pts : mHistory) {
                p1.x += pts.get(0).x;
                p1.y += pts.get(0).y;
                p2.x += pts.get(1).x;
                p2.y += pts.get(1).y;
                p3.x += pts.get(2).x;
                p3.y += pts.get(2).y;
                p4.x += pts.get(3).x;
                p4.y += pts.get(3).y;
            }
            p1.x /= mHistory.size()/3.;
            p1.y /= mHistory.size()/3.;
            p2.x /= mHistory.size()/3.;
            p2.y /= mHistory.size()/3.;
            p3.x /= mHistory.size()/3.;
            p3.y /= mHistory.size()/3.;
            p4.x /= mHistory.size()/3.;
            p4.y /= mHistory.size()/3.;

            return mPts;
        }
    }

    private List<Card> mCards;

    public Tracker() {
        mCards = new ArrayList<>();
    }

    private double intersection(Rect a, Rect b) {
        double x = Math.max(a.x, b.x);
        double y = Math.max(a.y, b.y);
        double w = Math.min(a.x+a.width, b.x+b.width) - x;
        double h = Math.min(a.y+a.height, b.y+b.height) - y;
        w = Math.max(0, w);
        h = Math.max(0, h);
        return w*h;
    }

    public void processDetections(List<List<Point>> dets) {
        for (List<Point> det : dets) {
            Rect det_rect = Imgproc.boundingRect(new MatOfPoint(det.get(0), det.get(1), det.get(2), det.get(3)));
            boolean card_already_tracked = false;
            for (Card card : mCards) {
                List<Point> card_pts = card.getPts();
                Rect card_rect = Imgproc.boundingRect(new MatOfPoint(card_pts.get(0), card_pts.get(1), card_pts.get(2), card_pts.get(3)));
                if (intersection(det_rect, card_rect) > 100) {
                    card_already_tracked = true;
                    card.detection(det);
                    break;
                }
            }
            if (card_already_tracked) continue;
            mCards.add(new Card(det, 5));
        }
    }

    public void drawCards(Mat frame) {
        List<MatOfPoint> pts = new ArrayList<>();
        for (Card card : mCards) {
            List<Point> card_pts = card.getPts();
            pts.add(new MatOfPoint(card_pts.get(0), card_pts.get(1), card_pts.get(2), card_pts.get(3)));
        }

        Imgproc.drawContours(frame, pts, -1, new Scalar(0,255,0), 3);


    }
}
