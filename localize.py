#! /usr/bin/python
import numpy as np
import cv2
import time

classify_width = 250
classify_height = 150

def order_points(pts):
    #pts = np.roll(pts, 1, axis=1)
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dist1 = np.linalg.norm(rect[0] - rect[1])
    dist2 = np.linalg.norm(rect[1] - rect[2])

    if dist1 < dist2:
        rect = np.roll(rect, 1, axis=0)
        

    return rect

def get_warp(points):
    rect = order_points(points).astype('float32')
    dest = np.array(
        [[0,0],
        [classify_width,0],
        [classify_width,classify_height],
        [0,classify_height]], dtype='float32')
    return cv2.getPerspectiveTransform(rect, dest)

def detection_candidates(frame):
    global binary

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    res, binary = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel, iterations = 2)
    binary_cpy = np.copy(binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if 1000 < cv2.contourArea(contour) < 20000]
    contours = [cv2.approxPolyDP(contour, 10, True) for contour in contours]
    contours = [contour for contour in contours if len(contour) == 4]
    return np.squeeze(contours)

def get_image(frame, points):
    warp = get_warp(points)
    warped = cv2.warpPerspective(frame, warp, (classify_width, classify_height))
    return warped

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  w = max(0, w)
  h = max(0, h)
  return w * h

binary = None
if __name__ == '__main__':
    detections = []
    cap = cv2.VideoCapture('one.mp4')
    flag = True
    while True:
        ret, frame = cap.read()

        # loop video
        #if frame is None:
        #    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #    continue

        # downsample
        height, width = frame.shape[:2] 
        scale = 0.5
        height = int(height*0.5)
        width = int(width*0.5)
        frame = cv2.resize(frame, (width, height))
        frame_cpy = np.copy(frame)

        # update trackers
        #for detection in detections:
        #    bbox, tracker = detection
        #    ok, new_bbox= tracker.update(frame)
        #    if not ok:
        #        detection[1] = None
        #    detection[0] = new_bbox
        #detections = [detection for detection in detections if detection[1] is not None]

        # find contours
        contours = detection_candidates(frame)

        #for contour in contours:
        #    img = get_image(frame, contour)
        #    t = int(time.time()*1000)
        #    cv2.imwrite('datagen/img%s.jpg' % t, img)

        # eventually filter candidates with svm

        # start tracking
        #for points in contours:
        #    rect1 = cv2.boundingRect(points)
        #    #print(rect1)
        #    already_tracking = False
        #    for rect2, tracker in detections:
        #        if intersection(rect1, rect2) > 100:
        #            already_tracking = True
        #            break
        #    if already_tracking:
        #        continue
        #    tracker = cv2.TrackerMOSSE_create()
        #    tracker.init(frame, rect1)
        #    detections.append([rect1, tracker])

        # draw result
        #for bbox, _ in detections:
        #    p1 = (int(bbox[0]), int(bbox[1]))
        #    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        if len(contours.shape) == 3:
            cv2.drawContours(frame_cpy, contours, -1, (0,255,0), 3)

        display = np.zeros((height*2, width, 3), dtype='uint8')
        display[:height,:,:] = frame_cpy
        display[height:,:,0] = binary





        cv2.imshow('frame', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
