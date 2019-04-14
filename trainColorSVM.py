import cv2
import sys
import numpy as np
import os

def img_hist(img) :
    mask = cv2.bitwise_not(cv2.inRange(img, np.array([150,150,150]), np.array([255,255,255])))
    masked_img = cv2.bitwise_and(img,img,mask=mask)
    hist = []
    for i in range(3) :
        hist.extend(cv2.calcHist([masked_img],[i],None,[8],[0,256]).flatten())
    return np.array(hist)


colors_dir = 'images/sorted/color' #sys.argv[1]
hists = []
labels = []

for c in os.listdir(colors_dir) :
    class_dir = os.path.join(colors_dir,c)
    if os.path.isdir(class_dir) :
        class_hists = [img_hist(cv2.imread(os.path.join(class_dir,image))).flatten() \
                      for image in os.listdir(class_dir)]
        class_labels = [c for x in class_hists]
        hists.extend(class_hists)
        labels.extend(class_labels)


label_to_int = {l:i for i,l in enumerate(np.unique(labels))}
int_to_label = {label_to_int[l]:l for l in label_to_int}
int_labels = np.array([label_to_int[x] for x in labels])
hists = np.array(hists)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(hists, cv2.ml.ROW_SAMPLE, int_labels)
svm.save('colorSVM.dat') #sys.argv[2])
