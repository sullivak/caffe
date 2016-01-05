import numpy as np
from PIL import Image
import os
import sys
import cv2
import glob

caffe_base_path = "/home/sullivan/Code/thirdparty/caffe-fullconv"
sys.path.insert(0, os.path.join(caffe_base_path, "python"))
import caffe

image_dir = "/home/sullivan/Dropbox (MC)/Data/ShivPics/Resized25"

output_dir = "/home/sullivan/Dropbox (MC)/Data/ShivPics/SegmentationTests/FCN2"

# http://stackoverflow.com/questions/32451934/image-per-pixel-scene-labeling-output-issue-using-fcn-32s-semantic-segmentation/32471602#32471602
caffemodel_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-32-60class", "fcn-32s-pascalcontext.caffemodel")
prototxt_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-32-60class", "deploy.prototxt")

#caffemodel_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-16", "fcn-16s-pascal.caffemodel")
#prototxt_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-16", "fcn-16s-pascal-deploy.prototxt")
#prototxt_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-16", "fcn-16s-pascal-deploy_updated.prototxt")

#caffemodel_path = os.path.join(caffe_base_path, "models/mcmodels/FCN/FCN-8", "fcn-8s-pascal.caffemodel")
#prototxt_path = os.path.join(caffe_base_path,  "models/mcmodels/FCN/FCN-8", "fcn-8s-pascal-deploy.prototxt")


image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
image_files.extend(glob.glob(os.path.join(image_dir, "*.JPG")))
image_files.extend(glob.glob(os.path.join(image_dir, "*.png")))

print len(image_files)

# load net
net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

for image_file_path in image_files:
    image_name = os.path.basename(image_file_path)
    image_name_base, image_name_extension = os.path.splitext(image_name)
    print "processing: " + image_name

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(image_file_path)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    #out = net.blobs['upscore'].data[0].argmax(axis=0) # <------ score or upscore
    mult = 1 # int(255.0 / 20.0) # <---- 20.0 or 59.0 or 255.0/just-plain-1
    out_path = os.path.join(output_dir, image_name_base + ".png")
    # print out_path
    cv2.imwrite(out_path, mult * out)


# cv2.imshow("out",out)
# cv2.waitKey(0)
# cv2.destroyWindow("out")
