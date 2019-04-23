import cv2
import sys
import os
import numpy as np
from CardClassifier import CardClassifier
# from svm import *
# from svmutil import *

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
img_dir = sys.argv[7]
out_dir = sys.argv[8]

card_dir = os.path.join(out_dir, 'card')
color_dir = os.path.join(out_dir, 'color')
shape_dir = os.path.join(out_dir, 'shape')
pattern_dir = os.path.join(out_dir, 'pattern')
number_dir = os.path.join(out_dir, 'number')

for x in cc.int_to_card :
    os.makedirs(os.path.join(card_dir, x), exist_ok=True)
for x in cc.int_to_color :
    os.makedirs(os.path.join(color_dir, x), exist_ok=True)
for x in cc.int_to_shape :
    os.makedirs(os.path.join(shape_dir, x), exist_ok=True)
for x in cc.int_to_pattern :
    os.makedirs(os.path.join(pattern_dir, x), exist_ok=True)
for x in cc.int_to_number :
    os.makedirs(os.path.join(number_dir, x), exist_ok=True)

ctr = 0
for filename in os.listdir(img_dir) :
    ctr += 1
    if ctr < 2000 :
        continue
    if ctr > 2500 :
        break
    img = cv2.imread(os.path.join(img_dir, filename))
    c = cc.predict(img)
    if c :
        pass
        cv2.imwrite(os.path.join(card_dir, cc.to_card(1), filename), img)
        cv2.imwrite(os.path.join(color_dir, cc.to_color(c[0]), filename), img)
        cv2.imwrite(os.path.join(number_dir, cc.to_number(c[1]), filename), img)
        cv2.imwrite(os.path.join(pattern_dir, cc.to_pattern(c[2]), filename), img)
        cv2.imwrite(os.path.join(shape_dir, cc.to_shape(c[3]), filename), img)
    else :
        pass
        cv2.imwrite(os.path.join(card_dir, cc.to_card(0), filename), img)
