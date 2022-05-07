from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cv2
 
if __name__ == '__main__':
    
    
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
      
       

        
        plt.imshow(img_path)
        plt.show()
        #print(img_path)  # e.g., ./static/img/xxx.jpg
       
        
    