import cv2
import os
import sys
import numpy as np

corners = []
cropped = False
true_x = 500
true_y = 300
true_corners = np.float32([[0,true_y],[0,0],[true_x,0],[true_x,true_y]])

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global cropped, corners
    # if the left mouse button was clicked, record the coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append([x, y])
        if len(corners) == 4 :
            cropped = True

def get_cropped(img) :
    global cropped, corners
    img = cv2.resize(img, (int(img.shape[1]/img.shape[0]*600), 600))
    failed = True
    while failed :
        failed = False
        corners = []
        cropped = False
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',click_and_crop)
        while(not cropped):
            cv2.imshow('image',img)
            cv2.waitKey(20)
        M = cv2.getPerspectiveTransform(np.float32(corners),true_corners)
        warp_img = cv2.warpPerspective(img,M,(true_x, true_y))
        cv2.imshow('image', warp_img)
        if chr(cv2.waitKey()) == 'x' :
            failed = True
    return warp_img

# args (in order) are:
# 1: card svm filename
# 2: color csv filename
# 3: number svm filename
# 4: pattern svm filename (ignored)
# 5: shape svm filename
# 6: hog descriptor filename
# 7: directory of input images
# 8: root directory of classified images
cc = CardClassifier(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]);

cam = cv2.VideoCapture(0)
while True :
    got_img = False
    while not got_img:
        ret, img = cam.read()
        cv2.imshow('image', img)
        if not ret:
            break
        k = cv2.waitKey(20)
        if k%256 == 32:
            # SPACE pressed
            got_img = True

    img = get_cropped(img)
    c = cc.predict(img)

    img = cv2.resize(img, (750, 450))
    if c :
        cv2.putText(img,cc.class_str(c),\
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2)
    else :
        cv2.putText(img,"NOT A CARD",\
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2)

    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k%256 == 27 :
        break
cam.release()
