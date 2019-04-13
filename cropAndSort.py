import cv2
import os
import sys
import numpy as np

corners = []
cropped = False
true_x = 500
true_y = 300
true_corners = np.float32([[0,0],[true_x,0],[true_x,true_y],[0,true_y]])

key_order = ['1','2','3']
key_to_color = {'1':'red', '2':'green', '3':'purple'}
key_to_shape = {'1':'squiggle', '2':'pill', '3':'diamond'}
key_to_pattern = {'1':'solid', '2':'empty', '3':'stripe'}
key_to_number = {'1':'1', '2':'2', '3':'3'}

unsorted_dir = sys.argv[1]
color_dir = os.path.join(unsorted_dir, os.pardir, 'sorted', 'color')
shape_dir = os.path.join(unsorted_dir, os.pardir, 'sorted', 'shape')
pattern_dir = os.path.join(unsorted_dir, os.pardir, 'sorted', 'pattern')
number_dir = os.path.join(unsorted_dir, os.pardir, 'sorted', 'number')

for x in key_to_color.values() :
    os.makedirs(os.path.join(color_dir, x), exist_ok=True)
for x in key_to_shape.values() :
    os.makedirs(os.path.join(shape_dir, x), exist_ok=True)
for x in key_to_pattern.values() :
    os.makedirs(os.path.join(pattern_dir, x), exist_ok=True)
for x in key_to_number.values() :
    os.makedirs(os.path.join(number_dir, x), exist_ok=True)


def choices_string(order, map) :
    str = ""
    for x in order[:-1] :
        str += x + ":" + map[x] + '    '
    str += order[-1] + ":" + map[order[-1]]
    return str

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropped
    # if the left mouse button was clicked, record the coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append([x, y])
        if len(corners) == 4 :
            cropped = True

for filename in os.listdir(unsorted_dir):
    img = cv2.imread(os.path.join(unsorted_dir,filename))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',click_and_crop)
    corners = []
    cropped = False
    while(not cropped):
        cv2.imshow('image',img)
        cv2.waitKey(20)
    cv2.destroyWindow('image')
    M = cv2.getPerspectiveTransform(np.float32(corners),true_corners)
    warp_img = cv2.warpPerspective(img,M,(true_x, true_y))

    color_img = np.copy(warp_img)
    cv2.putText(color_img,choices_string(key_order, key_to_color),\
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2)
    cv2.imshow('image',color_img)
    color = None
    while not color :
        key = chr(cv2.waitKey(0))
        if key in key_to_color :
            color = key_to_color[key]

    shape_img = np.copy(warp_img)
    cv2.putText(shape_img,choices_string(key_order, key_to_shape),\
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2)
    cv2.imshow('image',shape_img)
    shape = None
    while not shape :
        key = chr(cv2.waitKey(0))
        if key in key_to_shape :
            shape = key_to_shape[key]

    pattern_img = np.copy(warp_img)
    cv2.putText(pattern_img,choices_string(key_order, key_to_pattern),\
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2)
    cv2.imshow('image',pattern_img)
    pattern = None
    while not pattern :
        key = chr(cv2.waitKey(0))
        if key in key_to_pattern :
            pattern = key_to_pattern[key]

    number_img = np.copy(warp_img)
    cv2.putText(number_img,choices_string(key_order, key_to_number),\
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2)
    cv2.imshow('image',number_img)
    number = None
    while not number :
        key = chr(cv2.waitKey(0))
        if key in key_to_number :
            number = key_to_number[key]

    cv2.imwrite(os.path.join(color_dir, color, filename), warp_img);
    cv2.imwrite(os.path.join(shape_dir, shape, filename), warp_img);
    cv2.imwrite(os.path.join(pattern_dir, pattern, filename), warp_img);
    cv2.imwrite(os.path.join(number_dir, number, filename), warp_img);
