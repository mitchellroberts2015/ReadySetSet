import cv2
import numpy as np

class CardClassifier :
    def __init__(self, cardModel, colorModel, numberModel, patternModel, shapeModel, hogModel) :
        self.card_svm = cv2.ml.SVM_load(cardModel)
        self.true_colors = np.genfromtxt(colorModel, delimiter=",")
        self.number_svm = cv2.ml.SVM_load(numberModel)
        #self.pattern_svm = svm_load_model(sys.argv[4])
        self.shape_svm = cv2.ml.SVM_load(shapeModel)
        self.hog = cv2.HOGDescriptor(hogModel)

        self.int_to_card = ['negative', 'positive']
        self.int_to_color = ['green', 'purple', 'red']
        self.int_to_number = ['1', '2', '3']
        self.int_to_pattern = ['empty', 'solid', 'stripe']
        self.int_to_shape = ['diamond', 'pill', 'squiggle']


    def get_mean_color(self, img) :
        mask = cv2.bitwise_not(cv2.inRange(img, np.array([150,150,150]), np.array([255,255,255])))
        img = cv2.bitwise_and(img,img,mask=mask)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        return img[~np.all(img==0, axis=1)].mean(axis=0)

    def predict_color(self, img) :
        mean_color = self.get_mean_color(img)
        diffs = self.true_colors - mean_color
        norm_diffs = np.linalg.norm(diffs, axis=1)
        return np.argmin(norm_diffs)

    def predict_number(self, img, gray_img=None, img_hog=None) :
        if img_hog is None :
            if gray_img is None :
                gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_hog = self.hog.compute(gray_img).flatten()[np.newaxis]
        return int(self.number_svm.predict(img_hog)[1][0][0])

    def predict_pattern(self, img) :
        return 1;

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

    def class_to_str(c) :
        if c :
            return self.to_color(c[0]) + '    ' + self.to_number(c[1]) + '    ' + self.to_pattern(c[2]) + '    ' + self.to_shape(c[3])
        return "NOT A CARD"
