"""
Simple utility to see exact value of pixel hovering over. Needs mpldatacursor
Can provide filename as argument or leave blank for default
"""
import cv2
import mpldatacursor
import matplotlib.pyplot as plt
import sys

defaultFile = "/home/sullivan/Code/thirdparty/caffe-fullconv/python/test_out.png"
if len(sys.argv) < 2:
    imgPath = defaultFile
else:
    imgPath = sys.argv[1]

print imgPath
img = cv2.imread(imgPath)
fig, axes = plt.subplots()
axes.imshow(img)
mpldatacursor.datacursor(hover=True)
plt.show()
