import cv2
import sys
import os
import numpy as np
from CardClassifier import CardClassifier

# args (in order) are:
# 1: card svm filename
# 2: color csv filename
# 3: number svm filename
# 4: pattern svm filename (ignored)
# 5: shape svm filename
# 6: hog descriptor filename
cc = CardClassifier(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]);

for filename in os.listdir(img_dir) :
    img = cv2.imread(os.path.join(img_dir, filename))
    c = cc.predict(img)
    img = cv2.resize(img, (750, 450))
    if c :
        cv2.putText(img,cc.class_str(c),\
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2)
    else :
        cv2.putText(img,"NOT A CARD",\
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
