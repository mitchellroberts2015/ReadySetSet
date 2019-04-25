import cv2
import sys
import numpy as np
import os
from scipy.cluster.vq import vq, kmeans, whiten

def get_mean_color(img) :
    mask = cv2.bitwise_not(cv2.inRange(img, np.array([150,150,150]), np.array([255,255,255])))
    img = cv2.bitwise_and(img,img,mask=mask)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    return img[~np.all(img==0, axis=1)].mean(axis=0)

colors_dir = sys.argv[1]
true_colors = []

for c in os.listdir(colors_dir) :
    class_dir = os.path.join(colors_dir,c)
    if os.path.isdir(class_dir) :
        true_colors.append(np.mean([get_mean_color(cv2.imread(os.path.join(class_dir,image))) \
                      for image in os.listdir(class_dir)],0))

np.savetxt(sys.argv[2], np.array(true_colors), '%d', delimiter=',')
