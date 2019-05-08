import cv2
import sys
import os
import numpy as np
from CardClassifier import CardClassifier
from scipy.cluster.vq import vq, kmeans, whiten

classes_dir = sys.argv[1]
features = []
labels = []

def get_sats(img, block_size) :
    x_blocks = img.shape[1] // block_size[0]
    y_blocks = img.shape[0] // block_size[1]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sats = np.zeros((x_blocks, y_blocks, 2), np.float32)

    for y in range(y_blocks) :
        for x in range(x_blocks) :
            block = hsv_img[y*block_size[1]:(y+1)*block_size[1], x*block_size[0]:(x+1)*block_size[0],1]
            sats[x,y] = [np.mean(block), np.std(block)]

    return sats.flatten()

def predict_file(filename, svm) :
    f = get_features(img)
    c = svm.predict(f)
    return c[1][0][0]


for c in os.listdir(classes_dir) :
    if os.path.isdir(os.path.join(classes_dir,c)) :
        class_features = [get_sats(cv2.imread(os.path.join(classes_dir,c,image)), (25,25)) \
                      for image in os.listdir(os.path.join(classes_dir,c))]
        class_labels = [c for x in class_features]
        features.extend(class_features)
        labels.extend(class_labels)


label_to_int = {l:i for i,l in enumerate(np.unique(labels))}
int_to_label = {label_to_int[l]:l for l in label_to_int}
int_labels = np.array([label_to_int[x] for x in labels])
features = np.array(features)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(features, cv2.ml.ROW_SAMPLE, int_labels)
svm.save(sys.argv[2])
