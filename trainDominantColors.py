import os
import cv2
import numpy as np

true_colors = []

# Technically, we should use k-means or some other clustering algorithm to find
# the dominant colors in an image. However, the images in this case have only
# one color once the white pixels are filtered out, so we can just find the
# mean of the pixels (equivalent to doing k-means with k=1)
def get_mean_color(self, img) :
    mask = cv2.bitwise_not(cv2.inRange(img, np.array([150,150,150]), np.array([255,255,255])))
    img = cv2.bitwise_and(img,img,mask=mask)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    return img[~np.all(img==0, axis=1)].mean(axis=0)

colors_dir = 'images/sorted/color' #sys.argv[1]
for c in os.listdir(colors_dir) :
    class_dir = os.path.join(colors_dir,c)
    if os.path.isdir(class_dir) :
        mean_colors = np.array([get_mean_color(cv2.imread(os.path.join(class_dir,image))) \
                      for image in os.listdir(class_dir)])
        true_colors.append(np.mean(mean_colors, axis=0))

np.savetxt("colors.csv", np.array(true_colors), delimiter=",")
