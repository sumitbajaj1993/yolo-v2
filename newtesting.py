# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:46:34 2020

@author: KD823ER
"""

import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

#%config InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/tiny-yolo-voc-Copy.cfg',
    'load': 1800,
    'threshold': 0.01,
    #'gpu': 1.0
}

tfnet = TFNet(options)



img = cv2.imread('i91.jpg', cv2.IMREAD_COLOR)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img
# use YOLO to predict the image
result = tfnet.return_predict(img)

img.shape

tl = (result[200]['topleft']['x'], result[200]['topleft']['y'])
br = (result[233]['bottomright']['x'], result[233]['bottomright']['y'])
label = result[253]['label']


# add the box and label and display it
img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
plt.imshow(img)
plt.show()