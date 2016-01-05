import numpy as np
from PIL import Image
import os
import sys
import cv2


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# image_path = 'pascal/VOC2010/JPEGImages/2007_000129.jpg'
caffe_base_path = "/home/sullivan/Code/thirdparty/caffe-fullconv"
image_path = os.path.join(caffe_base_path, "data/whatever", "concorde.jpg") 
prototxt_path = os.path.join(caffe_base_path, "models/FCN", "deploy.prototxt")
caffemodel_path = os.path.join(caffe_base_path, "models/FCN", "fcn-32s-pascalcontext.caffemodel")

im = Image.open(image_path)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

preout = in_.astype(np.uint8)
#out = Image.fromarray(preout)
#out.save(os.path.join(caffe_base_path, "data/whatever", "transformed.tiff"))

