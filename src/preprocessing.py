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

# raw_image_path = '/home/lcc/code/data/SSRL_SKY_CAM_IMAGE/'
# input_data_path = '/home/lcc/code/python/SolarPrediction/dataset/NREL_SSRL_BMS_SKY_CAM/input_data/'
raw_image_path = '/media/lcc/Windows/Downloads/SSRL_SKY/'
input_data_path = '/home/lcc/code/python/SolarPrediction/dataset/NREL_SSRL_BMS_SKY_CAM/input_data/'
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

def pad_data(method, feat_size):
    new_data = []
    start_day = '20080101'
    end_day = '20160731'
    day_list = np.loadtxt('day_list.csv', dtype='str')
    date = np.loadtxt('exist_image_list.csv', dtype='str')
    feat = np.loadtxt(input_data_path + 'raw_' + method + '_' + str(feat_size) + '.csv', dtype='float')
    #data = np.loadtxt(path + method + '.csv', delimiter=',',dtype='str')
    #date = [str(int(float(i))) for i in data[:,0]]
    #feat = np.array(data[:,1:], dtype = 'float')
    #feat = data
    #feat_size = feat.shape[1]
    print feat
    print date
    print day_list
    idx = 0
    for day in day_list:
        hours = range(0,24)
        for hour in hours[:5]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([0] * feat_size)
        for hour in hours[5:-4]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day +  str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            print 'Day_Hour:', day_hour
            print 'Idx:', idx
            print date[idx]
            while date[idx] < day_hour:
                idx += 1
            if date[idx] == day_hour:
                new_data.append(feat[idx])
            else:
                f = np.array([-99999]*feat_size)
                new_data.append(f)
        for hour in hours[-4:]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([0] * feat_size)

    new_data = np.array(new_data)
    print new_data.shape
    np.savetxt(input_data_path + method + '_' + str(feat_size) + '.csv', new_data, fmt='%.4f',delimiter=',')

def pad_data_image_path():
    new_data = []
    start_day = '20080101'
    end_day = '20160731'
    day_list = np.loadtxt('day_list.csv', dtype='str')
    date = np.loadtxt('exist_image_list.csv', dtype='str')
    print date
    print day_list
    idx = 0
    for day in day_list:
        hours = range(0,24)
        for hour in hours[:5]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([-11111])
        for hour in hours[5:-4]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day +  str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            print 'Day_Hour:', day_hour
            print 'Idx:', idx
            print date[idx]
            while date[idx] < day_hour:
                idx += 1
            if date[idx] == day_hour:
                new_data.append([int(date[idx])])
            else:
                new_data.append([-99999])
        for hour in hours[-4:]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([-11111])

    new_data = np.array(new_data)
    print new_data.shape
    np.savetxt('pad_data_path.csv', new_data, fmt='%12.0f', delimiter=',')
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
#         print raw_image_path + str(y) + '/'
#         feat_list = []
#         for parent, dirnames, filenames in os.walk(raw_image_path+str(y)+'/'):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
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
#         np.savetxt(input_data_path+str(y)+"_rcnnet.csv", feat_list, delimiter=",")

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
        print raw_image_path + str(y) + '/'
    #    feat_list = []
        for parent, dirnames, filenames in os.walk(raw_image_path + str(y) + '/'):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
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
                elif method == 'LBP':
                    feat = LBP_feature(img)
                elif method == 'DSIFT':
                    feat = Dense_sift(img)
                elif method == 'exist_image':
                    feat = int(filename[:-4])
                feat_list.append(feat)
    feat_list = np.array(feat_list)
    print feat_list.shape

    #np.savetxt(input_data_path + 'raw_' + method + '_' + str(feat_list.shape[1]) + '.csv', feat_list, delimiter=",", fmt = '%.4f')
    #np.savetxt('input_data_path.csv', feat_list, delimiter=',',fmt = '%12.0f')
def all_sky_image_features(img_lst):
    res = []
    img_lst_gray = []
    for img in img_lst:
        print type(img), img.shape
        img_lst_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    '''
     Numbers of cloud pixels
    '''
    NC = []; Thr_NC = 0.8
    for img in img_lst:
        b, g, r = cv2.split(img)
        RBR = np.true_divide(r, b + 0.001)
        NC.append(np.sum([1 if i > Thr_NC else 0 for i in RBR.ravel()]))
    res.append(np.mean(NC))
    res.append(np.var(NC))
    #print 'mean(NC) var(NC)', np.mean(NC), np.var(NC)

    '''
    Frame difference
    '''
    DA = []; DC = []; Thr_DF = 0.1
    for i in range(len(img_lst)):
        if i == 0: continue
        DA.append([abs(i) if abs(i) > Thr_DF else 0 for i in (img_lst[i].ravel() - img_lst[i - 1].ravel())])
        DC.append([1 if abs(i) > Thr_DF else 0 for i in (img_lst[i].ravel() - img_lst[i - 1].ravel())])
    res.append(np.mean(DA))
    res.append(np.mean(DC))
    res.append(np.var(DA))
    res.append(np.var(DC))
    #print 'mean(DA) var(DA)', np.mean(DA), np.var(DA)
    #print 'mean(DC) var(DC)', np.mean(DC), np.var(DC)

    '''
    Gradient magnitude
    '''
    GM = []
    for img in img_lst_gray:
        dst = cv2.Sobel(img, -1, 1, 1)
        GM.append(np.sum(dst.ravel()))
    res.append(np.mean(GM))
    res.append(np.var(GM))
    #print 'mean(GM) var(GM)', np.mean(GM), np.var(GM)

    '''
    Intensity Level
    '''
    IL = []
    for img in img_lst_gray:
        IL.append(np.mean(img.ravel()))
    res.append(np.mean(IL))
    res.append(np.var(IL))
    #print 'mean(IL) var(IL)', np.mean(IL), np.var(IL)

    '''
    Accumulated intensity along the vertical line of sum
    '''
    # for img in img_lst_gray:
    #     dst = (cv2.HoughLines(img, 1, math.pi / 180, 125))[0]
    #     ang = [i if i >= 0 and i <= math.pi / 2 else abs(math.pi - i) for i in dst[:,1]]
    #     ndst = np.array([i if i[1] < 1 else None for i in dst])
    #     idx = np.argsort(ang)
    #     print ndst
    #     #print dst[idx]

    '''
    Number of Corners
    '''
    COR = []
    for img in img_lst_gray:
        dst = cv2.cornerHarris(img, 7, 7, 0.04)
        COR.append(np.sum([1 if i > (0.1 * dst.max()) else 0 for i in dst.ravel()]))
    res.append(np.mean(COR))
    res.append(np.var(COR))
    #print 'mean(COR) var(COR)', np.mean(COR), np.var(COR)

    return res

if __name__== '__main__':
    pad_data_image_path()
    #pad_data()
   #get_feature('exist_image')
   # img = cv2.imread(raw_image_path + '2016/01/01/201601011000.jpg')
   # fea = Dense_sift(img)
   # print len(fea)
