import cv2
import sys
import os
import numpy as np

classes_dir = sys.argv[1]
hogs = []
labels = []
hog = cv2.HOGDescriptor((64,128), (32,32), (16,16), (16,16), 9)

def predict_file(filename, hog, svm, map) :
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hog = hog.compute(img).flatten()[np.newaxis]
    c = svm.predict(img_hog)
    return map[c[1][0][0]]

for c in os.listdir(classes_dir) :
    if os.path.isdir(os.path.join(classes_dir,c)) :
        class_hogs = [hog.compute(cv2.imread(os.path.join(classes_dir,c,image))).flatten() \
                      for image in os.listdir(os.path.join(classes_dir,c))]
        class_labels = [c for x in class_hogs]
        hogs.extend(class_hogs)
        labels.extend(class_labels)


label_to_int = {l:i for i,l in enumerate(np.unique(labels))}
int_to_label = {label_to_int[l]:l for l in label_to_int}
int_labels = np.array([label_to_int[x] for x in labels])
hogs = np.array(hogs)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(hogs, cv2.ml.ROW_SAMPLE, int_labels)
svm.save(sys.argv[2])
hog.save(sys.argv[3])
