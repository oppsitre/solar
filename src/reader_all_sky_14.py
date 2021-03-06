# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import cv2
import random
from preprocessing import all_sky_image_features
#import sys
#sys.path.append('/usr/local/lib/python2.7/dist-packages')

HOUR_IN_A_DAY = 24
MISSING_VALUE = -99999

sky_cam_train_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/train/sky_cam_train_data.csv"
target_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"

sky_cam_validation_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/validation/sky_cam_validation_data.csv"
target_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"

sky_cam_test_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/test/sky_cam_test_data.csv"
target_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"

#sky_cam_raw_data_path = '../dataset/NREL_SSRL_BMS_SKY_CAM/SSRL_SKY/'
sky_cam_raw_data_path = '../dataset/NREL_SSRL_BMS_SKY_CAM/SSRL_SKY/'

class Reader:

    def _feature_reshape(self, features, data_step, n_step):
        """
        @brief aggregate the multiple features (a successive time features) as a new feature to predict the following time target
        @param features The input features in shape [features_num, feature_dim], each of which is a feature of a specific time
        @param data_step The interval between two new features
        @param n_step The length of the time in the lstm
        @return a new batch shaped features in shape [new_feature_num, n_step, feature_dim]
        """
        shape_features = []
        for ptr in range(n_step, len(features), data_step):
            shape_features.append(features[ptr - n_step:ptr])
        return np.array(shape_features)

    def _target_reshape(self, targets, data_step, n_step, h_ahead, n_target):
        """
        @brief aggregate the multiple features as a new target for multiple output and synchronize the target with the features
        @param targets The input targets in shape [target_num, target_dim] ##now target_num=1
        @param n_step  The interval between two new features
        @param h_ahead The interval of the feature and target
        @param n_target The  number of target for a prediction
        @return an aggregated and synchronized targets in shape [new_target_num, n_target, target_dim]
        """
        shape_targets = []
        for ptr in range(n_step + h_ahead + n_target, len(targets), data_step):
            shape_targets.append(targets[ptr - n_target:ptr])
        return np.array(shape_targets)

    def _get_valid_index(self, sky_cam_features, targets):
        """
        @brief get the valida index of the features since there are some missing value in some feature and target
        @param sky_cam_features: the feautures
        @return Return a indices indicates the valid features (no missing value) index
        """
        num = len(targets)
        # print num
        missing_index = []
        for i in range(len(targets)):
            if (MISSING_VALUE in sky_cam_features[i]) or \
                (True in np.isnan(sky_cam_features[i])) or \
                (MISSING_VALUE in targets[i]):
                missing_index.append(i)
        print 'Missing_index:', missing_index
        return np.setdiff1d(np.arange(num), np.array(missing_index))

    def __init__(self, config):
        """
        The constructor of the class Reader
        load the train, validation and test data from the dataset and call the function to aggregate and synchronize the features and target
        filter the some points with the missing value
        and do the feature pre-processing (now just scale the feature into mean is 0 and stddev is 1.0)
        """

        #load data
        sky_cam_train_raw_data = np.array(np.loadtxt(sky_cam_train_data_path, delimiter=',', dtype='float'), dtype='int')
        sky_cam_validation_raw_data = np.array(np.loadtxt(sky_cam_validation_data_path, delimiter=',', dtype='float'), dtype='int')
        sky_cam_test_raw_data = np.array(np.loadtxt(sky_cam_test_data_path, delimiter=',', dtype='float'), dtype='int')

        # sky_cam_train_raw_data = np.array(sky_cam_train_raw_data, dtype='int')
        # print sky_cam_train_raw_data

        target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
        target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
        target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

        #feature eshape
        #feature reshape: accumulate several(n_step) features into a new feature for the input the lstm
        #target reshape: align the target with the input feature

        self.sky_cam_train_data = self._feature_reshape(sky_cam_train_raw_data, config.data_step, config.n_step)
        #print 'sky_cam_train_data:', self.sky_cam_train_data.dtype, self.sky_cam_train_data
        self.sky_cam_validation_data = self._feature_reshape(sky_cam_validation_raw_data, config.data_step, config.n_step)
        #print 'sky_cam_validation_data:', self.sky_cam_validation_data.dtype, self.sky_cam_validation_data
        self.sky_cam_test_data = self._feature_reshape(sky_cam_test_raw_data, config.data_step, config.n_step)
        #print 'sky_cam_test_data:', type(self.sky_cam_test_data), self.sky_cam_test_data

        self.target_train_data = self._target_reshape(target_train_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_validation_data = self._target_reshape(target_validation_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_test_data = self._target_reshape(target_test_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)

        self.train_index = self._get_valid_index(self.sky_cam_train_data, self.target_train_data)
        #print 'self.train_index:', self.train_index
        self.validation_index = self._get_valid_index(self.sky_cam_validation_data, self.target_validation_data)
        #print 'validation_index:', self.validation_index
        self.test_index = self._get_valid_index(self.sky_cam_test_data, self.target_test_data)
        #print 'self.test_index:', self.test_index
        self.n_step = config.n_step
        self.width = config.width
        self.heigth = config.heigth
        #concatenate all valid data
        sky_cam_raw_valid_data = np.concatenate((sky_cam_train_raw_data[self.train_index], sky_cam_validation_raw_data[self.validation_index], sky_cam_test_raw_data[self.test_index]), axis=0)

        #feature scale
        # sky_cam_mean = np.mean(sky_cam_raw_valid_data, axis=0)
        # sky_cam_std = np.std(sky_cam_raw_valid_data, axis=0)
        # self.sky_cam_train_data = (self.sky_cam_train_data - sky_cam_mean) / sky_cam_std
        # self.sky_cam_validation_data = (self.sky_cam_validation_data - sky_cam_mean) / sky_cam_std
        # self.sky_cam_test_data = (self.sky_cam_test_data - sky_cam_mean) / sky_cam_std

        #CAUTIOUS: the length of the ir_tarin_data and target_train_data may be differnet
        #the length of mete_test_data may be more short
        #and thus we must use the target data to compute the number
        self.train_num = len(self.target_train_data)
        self.validataion_num = len(self.target_validation_data)
        self.test_num = len(self.target_test_data)

        self.batch_size = config.batch_size

        self.noise_mean = config.noise_mean
        self.noise_vari = config.noise_vari
        self.noise_ratio = config.noise_ratio
        # self.index = np.random.random_integers(0, self.train_num-1, size=(self.batch_size))
        #print the dataset info
        print "Dataset info"
        print "="*80
        print "train number:", self.train_num
        print "validation number:", self.validataion_num
        print "test number", self.test_num
        print "batch size:", self.batch_size
        print "use", config.n_step, "hours to predict the next ", config.n_target, " consecutive hours"
        print "\n\n"


    def path2image(self, data, index, add_noise = False, image_scale = False):
        mean = cv2.resize(np.load('mean.npy'), (self.heigth, self.width))
        std = cv2.resize(np.load('std.npy'), (self.heigth, self.width))
        img_list = []
        for idx in index:
            imgs = []
            for i in range(self.n_step):
                if data[idx, i] == -11111:
                    imgs.append(np.zeros((self.heigth,self.width, 3), dtype='uint8'))
                else:
                    filename = str(int(data[idx, i]))
                    y = filename[:4]
                    m = filename[4:6]
                    d = filename[6:8]
                    path = sky_cam_raw_data_path + str(y) + '/' + str(m) + '/' + str(d) + '/' + str(filename) + '.jpg'
                    #tt = cv2.imread(path)
                    #print i, filename, tt.dtype
                    #cv2.imshow(filename, tt)
                    #cv2.waitKey(0)
                    tmp = cv2.resize(cv2.imread(path), (self.heigth, self.width))
                    # if image_scale is True:
                    #     tmp -= mean
                    #     tmp /= std
                    # mean_noise = self.noise_mean
                    # sigma_noise = self.noise_vari
                    # noise = np.random.normal(mean_noise, sigma_noise, size=(self.heigth, self.width))
                    # if add_noise is True and random.random() < self.noise_ratio:
                    #     tmp += noise
                    imgs.append(tmp)
            img_list.append(imgs)
        return np.array(img_list)

    def next_batch(self):
        """
        @brief return a batch of train and target data
        @return ir_data_batch: [batch_size, n_step, n_input]
        @return mete_data_batch:  [batch_size, n_step, n_input]
        @return target_data_batch: [n_model, batch_size, n_target]
        """
        index = np.random.choice(self.train_index, self.batch_size)
        batch_images = self.path2image(self.sky_cam_train_data, index)
        sky_cam_batch_data = []
        for batch in batch_images:
            sky_cam_batch_data.append(all_sky_image_features(batch))
        sky_cam_batch_data = np.array(sky_cam_batch_data)
        #print '!!!!!!!!!!!!!!!!!!!!!!sky_cam_batch_data.shape:', sky_cam_batch_data.shape
        target_batch_data = self.target_train_data[index]

        return sky_cam_batch_data, \
                target_batch_data

    def get_train_set(self):
        """
        @brief return the total dataset
        """
        #index = np.random.choice(self.validation_index, self.)
        return self.sky_cam_train_data[self.train_index], \
                self.target_train_data[self.train_index]

    #The returned validataion and test set:
    #ir_data and mete_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    def get_validation_set(self):
        """
        @brief return the total validation dataset
        """
        index = np.random.choice(self.validation_index, self.validataion_num)
        batch_images = self.path2image(self.sky_cam_validation_data, index)
        sky_cam_validation_batch_data = []
        for batch in batch_images:
            sky_cam_validation_batch_data.append(all_sky_image_features(batch))
        sky_cam_validation_batch_data = np.array(sky_cam_validation_batch_data)


        return sky_cam_validation_batch_data, self.target_validation_data[0:self.validataion_num]
        # return self.sky_cam_validation_data[0:self.validataion_num], \
        #         self.target_validation_data[0:self.validataion_num]

    def get_test_set(self, test_num):
        """
        @brief return a test set in the specific test num
        @param test_num The number of test set to return
        """
        #a = self.path2image(self.sky_cam_test_data, range(test_num))
        index = np.random.choice(self.test_index, test_num)
        batch_images = self.path2image(self.sky_cam_test_data, index)
        sky_cam_test_batch_data = []
        for batch in batch_images:
            sky_cam_test_batch_data.append(all_sky_image_features(batch))
        sky_cam_test_batch_data = np.array(sky_cam_test_batch_data)
        #print 'AAAAAAAAAAAAAAAAAAAAAA.shape', a.shape
        return sky_cam_test_batch_data, self.target_test_data[0:test_num]
        # return self.sky_cam_test_data[0:test_num], \
        #         self.target_test_data[0:test_num]
