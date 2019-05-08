import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

class CardClassifier :
    def __init__(self, cardModel, colorModel, numberModel, patternModel, shapeModel, hogModel) :
        self.card_svm = cv2.ml.SVM_load(cardModel)
        self.true_colors = np.genfromtxt(colorModel, delimiter=",")
        self.true_hues = np.apply_along_axis(self.bgr_hue, 1, self.true_colors)
        self.number_svm = cv2.ml.SVM_load(numberModel)
        self.pattern_svm = cv2.ml.SVM_load(patternModel)
        self.shape_svm = cv2.ml.SVM_load(shapeModel)
        self.hog = cv2.HOGDescriptor(hogModel)

        self.int_to_card = ['negative', 'positive']
        self.int_to_color = ['green', 'purple', 'red']
        self.int_to_number = ['1', '2', '3']
        self.int_to_pattern = ['empty', 'solid', 'stripe']
        self.int_to_shape = ['diamond', 'pill', 'squiggle']

    def bgr_hue(self, bgr) :
        return cv2.cvtColor(np.array([[bgr]],np.float32),cv2.COLOR_BGR2HSV)[0,0,0]

    def get_mean_color(self, img) :
        mask = cv2.bitwise_not(cv2.inRange(img, np.array([150,150,150]), np.array([255,255,255])))
        img = cv2.bitwise_and(img,img,mask=mask)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        return img[~np.all(img==0, axis=1)].mean(axis=0)

    def get_sats(self, img, block_size) :
        x_blocks = img.shape[1] // block_size[0]
        y_blocks = img.shape[0] // block_size[1]
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sats = np.zeros((x_blocks, y_blocks, 2), np.float32)

        for y in range(y_blocks) :
            for x in range(x_blocks) :
                block = hsv_img[y*block_size[1]:(y+1)*block_size[1], x*block_size[0]:(x+1)*block_size[0],1]
                sats[x,y] = [np.mean(block), np.std(block)]

        return sats.flatten()

    def predict_color(self, img) :
        color = self.get_mean_color(img)
        diffs = np.abs(self.true_hues - self.bgr_hue(color))
        diffs[diffs>180] = -1 * diffs[diffs>180] + 360
        return np.argmin(diffs)

    def predict_number(self, img, gray_img=None, img_hog=None) :
        if img_hog is None :
            if gray_img is None :
                gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_hog = self.hog.compute(gray_img).flatten()[np.newaxis]
        return int(self.number_svm.predict(img_hog)[1][0][0])

    def predict_pattern(self, img) :
        sats = self.get_sats(img, (25,25))[np.newaxis]
        return int(self.pattern_svm.predict(sats)[1][0][0])

    def predict_shape(self, img, gray_img=None, img_hog=None) :
        if img_hog is None :
            if gray_img is None :
                gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_hog = self.hog.compute(gray_img).flatten()[np.newaxis]
        return int(self.shape_svm.predict(img_hog)[1][0][0])

    def predict(self, img) :
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_hog = self.hog.compute(gray_img).flatten()[np.newaxis]
        card = self.card_svm.predict(img_hog)[1][0][0]
        if card == 0 :
            return None

        color = self.predict_color(img)
        number = self.predict_number(img, img_hog=img_hog)
        pattern = self.predict_pattern(img)
        shape = self.predict_shape(img, img_hog=img_hog)

        return (color, number, pattern, shape)

    def to_card(self, i) :
        return self.int_to_card[i]

    def to_color(self, i) :
        return self.int_to_color[i]

    def to_number(self, i) :
        return self.int_to_number[i]

    def to_pattern(self, i) :
        return self.int_to_pattern[i]

    def to_shape(self, i) :
        return self.int_to_shape[i]

    def class_to_str(self,c) :
        if c :
            return self.to_color(c[0]) + '    ' + self.to_number(c[1]) + '    ' + self.to_pattern(c[2]) + '    ' + self.to_shape(c[3])
        return "NOT A CARD"
