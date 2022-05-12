import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import cv2


import pywt
app = Flask(__name__, template_folder='./')
def w2d(img, mode='haar', level=1):
        imArray = img
    #Datatype conversions
    #convert to grayscale
        imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
        imArray =  np.float32(imArray)   
        imArray /= 255;
    # compute coefficients 
        coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
        coeffs_H=list(coeffs)  
        coeffs_H[0] *= 0;  

    # reconstruction
        imArray_H=pywt.waverec2(coeffs_H, mode);
        imArray_H *= 255;
        imArray_H =  np.uint8(imArray_H)

        return imArray_H
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = np.array(Image.open(file.stream))  # PIL image
        scalled_raw_img3 = cv2.resize(img, (32, 32))
        #uploaded_img_path = "C:/Users/roni/Desktop/TA_project/static/upload" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        #img.save(uploaded_img_path)
        
        # Run search
        query = w2d(img,'db1',5)
        scalled_img_har1 = cv2.resize(query, (32, 32))
        combined_img3 = np.vstack((scalled_raw_img3.reshape(32*32*3,1),scalled_img_har1.reshape(32*32,1)))


        len_image_array = 32*32*3 + 32*32
        
        final = combined_img3.reshape(1,len_image_array).astype(float)
        dists = np.linalg.norm(features-final, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:2]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',scores=scores)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()