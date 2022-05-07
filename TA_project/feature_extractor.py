import cv2
from keras.applications.vgg16 import VGG16  , preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pywt
import matplotlib as plt

import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        """
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        """
        self.pipe = SVC()
    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)
        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x , y = preprocess_input(x)  # Subtracting avg values for each pixel
        x = self.pipe.fit(x,y)
        feature = self.pipe.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize

    def w2d(img, mode='haar', level=1):
        imArray = img
        imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
        imArray = np.float32(imArray)
        imArray /= 255;
        coeffs = pywt.wavedec2(imArray, mode , level=level)

        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0;
        imArray_H = pywt.waverec2(coeffs_H, mode);
        imArray_H *= 255;
        imArray = np.uint8(imArray_H)

        return imArray_H



    