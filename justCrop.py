import cv2
import os
import sys
import numpy as np

corners = []
cropped = False
true_x = 500
true_y = 300
true_corners = np.float32([[0,true_y],[0,0],[true_x,0],[true_x,true_y]])

raw_dir = sys.argv[1]
crop_dir = sys.argv[2]
os.makedirs(crop_dir, exist_ok=True)

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropped
    # if the left mouse button was clicked, record the coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append([x, y])
        if len(corners) == 4 :
            cropped = True

for filename in os.listdir(raw_dir):
    img = cv2.imread(os.path.join(raw_dir,filename))
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
        cv2.destroyWindow('image')
        M = cv2.getPerspectiveTransform(np.float32(corners),true_corners)
        warp_img = cv2.warpPerspective(img,M,(true_x, true_y))
        cv2.imshow('image', warp_img)
        if chr(cv2.waitKey()) == 'x' :
            failed = True
    warp_img = cv2.resize(warp_img, (true_x//2, true_y//2))
    cv2.imwrite(os.path.join(crop_dir, filename), warp_img);
