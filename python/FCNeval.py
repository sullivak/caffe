import numpy as np
from PIL import Image
import os
import sys
import cv2

caffe_base_path = "/home/sullivan/Code/thirdparty/caffe-fullconv"
sys.path.insert(0, os.path.join(caffe_base_path, "python"))
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# image_path = 'pascal/VOC2010/JPEGImages/2007_000129.jpg'
# image_path = os.path.join(caffe_base_path, "data/FCNtests", "concorde.jpg") 
# image_path = os.path.join(caffe_base_path, "data/FCNtests", "messyOffice.jpg") 
# image_path = os.path.join(caffe_base_path, "data/FCNtests", "train_25.jpg") 
image_path =  "/home/sullivan/Dropbox (MC)/MC Team Folder/Research/ImageClassification/caffe_exps_data/data/aerial/images/houses_025.jpg" 

# http://stackoverflow.com/questions/32451934/image-per-pixel-scene-labeling-output-issue-using-fcn-32s-semantic-segmentation/32471602#32471602
caffemodel_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-32-60class", "fcn-32s-pascalcontext.caffemodel")
prototxt_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-32-60class", "deploy.prototxt")

#caffemodel_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-16", "fcn-16s-pascal.caffemodel")
#prototxt_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-16", "fcn-16s-pascal-deploy.prototxt")
#prototxt_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-16", "fcn-16s-pascal-deploy_updated.prototxt")

#caffemodel_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-8", "fcn-8s-pascal.caffemodel")
#prototxt_path = os.path.join(caffe_base_path,  "models/mcmodels/FCN/FCN-8", "fcn-8s-pascal-deploy.prototxt")

# load net
net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

im = Image.open(image_path)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
#out = net.blobs['score'].data[0].argmax(axis=0)
out = net.blobs['upscore'].data[0].argmax(axis=0) # <------ score or upscore
mult = int(255.0 / 20.0) # <---- 20.0 or 59.0 or 255.0
cv2.imwrite("test_out.png", mult * out)


# cv2.imshow("out",out)
# cv2.waitKey(0)
# cv2.destroyWindow("out")
