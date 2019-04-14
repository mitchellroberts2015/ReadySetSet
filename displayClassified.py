import cv2
import sys
import os
import numpy as np

card_svm = cv2.ml.SVM_load(sys.argv[1])
color_svm = cv2.ml.SVM_load(sys.argv[2])
number_svm = cv2.ml.SVM_load(sys.argv[3])
pattern_svm = cv2.ml.SVM_load(sys.argv[4])
shape_svm = cv2.ml.SVM_load(sys.argv[5])
hog = cv2.HOGDescriptor(sys.argv[6])
img_dir = sys.argv[7]

int_to_card = ['negative', 'positive']
int_to_color = ['green', 'purple', 'red']
int_to_number = ['1', '2', '3']
int_to_pattern = ['empty', 'solid', 'stripe']
int_to_shape = ['diamond', 'pill', 'squiggle']

def class_str(c) :
    str = ''
    str += 'color: ' + int_to_color[int(c[0])]
    str += '    number: ' + int_to_number[int(c[1])]
    str += '    pattern: ' + int_to_pattern[int(c[2])]
    str += '    shape: ' + int_to_shape[int(c[3])]
    return str

def predict_img(img) :
    global card_svm, color_svm, number_svm, pattern_svm, shape_svm, hog
    img_hog = hog.compute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)).flatten()[np.newaxis]
    if card_svm.predict(img_hog)[1][0][0] == 0 :
        return None
    img_color_hist = img_hist(img)[np.newaxis]
    color = color_svm.predict(img_color_hist)[1][0][0]
    number = number_svm.predict(img_hog)[1][0][0]
    pattern = pattern_svm.predict(img_hog)[1][0][0]
    shape = shape_svm.predict(img_hog)[1][0][0]
    return (color, number, pattern, shape)


def img_hist(img) :
    mask = cv2.bitwise_not(cv2.inRange(img, np.array([150,150,150]), np.array([255,255,255])))
    masked_img = cv2.bitwise_and(img,img,mask=mask)
    hist = []
    for i in range(3) :
        hist.extend(cv2.calcHist([masked_img],[i],None,[8],[0,256]).flatten())
    return np.array(hist)

for filename in os.listdir(img_dir) :
    img = cv2.imread(os.path.join(img_dir, filename))
    c = predict_img(img)
    img = cv2.resize(img, (750, 450))
    if c :
        cv2.putText(img,class_str(c),\
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2)
    else :
        cv2.putText(img,"NOT A CARD",\
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
