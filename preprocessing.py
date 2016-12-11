#-*- coding: utf-8 -*-
'''
According to the paper, the authors extracted upto 80 frames from each video,
they did not mention if they grabbed first 80 frames, or sampled 80 frames with same intervals,
but anyway I did the latter.
'''
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import os
import ipdb
import numpy as np
import pandas as pd
from skimage import feature
#from cnn_util import *

image_load_path = '/home/lcc/code/data/SSRL_SKY_CAM_IMAGE/'
image_save_path = '/home/lcc/code/python/SolarPrediction/dataset/NREL_SSRL_BMS_SKY_CAM/input_data/'
year = range(1999, 2017)
def preprocess_frame(image, target_height=224, target_width=224):

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

# def prerain_CNN():
#     vgg_model = '/home/lcc/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'
#     vgg_deploy = '/home/lcc/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'
#     year = range(1999,2015)
#
#     #videos = filter(lambda x: x.endswith('avi'), videos)
#     width = 227
#     height = 227
#     print 'Before CNN'
#     cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=width, height=height)
#     print 'After CNN'
#     for y in year[::-1]:
#         print image_load_path + str(y) + '/'
#         feat_list = []
#         for parent, dirnames, filenames in os.walk(image_load_path+str(y)+'/'):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
#             for filename in filenames:  # 输出文件信息
#                 if not filename.endswith('.jpg'):
#                     continue
#                 # print "parent is:" + parent
#                 # print "filename is:" + filename
#                 # print "the full name of the file is:" + os.path.join(parent, filename)  # 输出文件路径信息
#                 file = os.path.join(parent, filename)
#                 img_list = []
#                 img = cv2.imread(file)
#                 if img is None:
#                     continue
#                 img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
#                 img_list.append(img)
#                 img_list = np.array(img_list)
#                 feat = cnn.get_features(img_list,layers='fc7',layer_sizes=[4096])[0].tolist()
#                 print filename[:-4]
#                 feat.insert(0, int(filename[:-4]))
#                 feat_list.append(feat)
#         feat_list = np.array(feat_list)
#         print feat_list.shape
#         np.savetxt(image_save_path+str(y)+"_rcnnet.csv", feat_list, delimiter=",")

def Dense_sift(img):
    img = cv2.resize(img,(25,25))
    #dense = cv2.FeatureDetector_create('Dense')
    dense = cv2.FeatureDetector_create('Dense')
    f = '{} ({}): {}'
    for param in dense.getParams():
        type_ = dense.paramType(param)
        if type_ == cv2.PARAM_BOOLEAN:
            print f.format(param, 'boolean', dense.getBool(param))
        elif type_ == cv2.PARAM_INT:
            print f.format(param, 'int', dense.getInt(param))
        elif type_ == cv2.PARAM_REAL:
            print f.format(param, 'real', dense.getDouble(param))
        else:
            print param
    #dense = cv2.setDouble('')
    dense.setDouble('initFeatureScale', 10)
    dense.setDouble('featureScaleMul', 10)
    dense.setInt('initXyStep', 10)
    print type(dense)
    kp = dense.detect(img)
    print len(kp)
    sift = cv2.SIFT()
    kp,des = sift.compute(img,kp)
    print des.ravel().shape
    return des
    #print type(kp[0]), type(des[0])
    #print len(kp), len(des)

def HOG_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray,(64,128))
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    h = h.ravel()
    return h.tolist()


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist
def LBP_feature(img, numPoints = 24, radius = 8):
    desc = LocalBinaryPatterns(numPoints, radius)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    #print hist.shape
    return hist.tolist()
def get_feature(method):
    feat_list = []
    for y in year:
        print image_load_path + str(y) + '/'
    #    feat_list = []
        for parent, dirnames, filenames in os.walk(image_load_path + str(y) + '/'):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
            for filename in filenames:  # 输出文件信息
                if not filename.endswith('.jpg'):
                    continue
                file = os.path.join(parent, filename)
                img = cv2.imread(file)
                if img is None:
                    continue
                print filename[:-4]
                if method == 'HOG':
                    feat = HOG_feature(img)
		    print type(feat)
		    print len(feat)
                elif method == 'LBP':
                    feat = LBP_feature(img)
                elif method == 'DSIFT':
                    feat = Dense_sift(img)
                #feat.insert(0, int(filename[:-4]))
                feat_list.append(feat)
    feat_list = np.array(feat_list)
    print feat_list.shape
    np.savetxt(image_save_path + 'raw_' + method + '_' + str(feat_list.shape[1]) + '.csv', feat_list, delimiter=",")
if __name__=="__main__":
    get_feature('HOG')
   # img = cv2.imread(image_load_path + '2016/01/01/201601011000.jpg')
   # fea = Dense_sift(img)
   # print len(fea)
