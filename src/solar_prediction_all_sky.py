# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import tensorflow as tf
import numpy as np
import json
from collections import namedtuple
from reader_all_sky_14 import Reader
from model_all_sky_14 import Model
from util import MSE_And_MAE, test_figure_plot
import sys
import datetime
import time
t = time.strftime('%m%d%H%M',time.localtime(time.time()))
#log_file = open('all_sky_log_'+ t,'w')
#sys.stdout = log_file

def main():
    #get the config
    print time.localtime()
    fp = open('../config.json')
    config = json.load(fp, object_hook=lambda d:namedtuple('X', d.keys())(*d.values()))
    fp.close()

    n_step = config.n_step
    n_target = config.n_target
    width_image = config.width
    height_image = config.heigth
    epoch_size = config.epoch_size
    print_step = config.print_step

    test_num = config.test_num

    #define the input and output
    x_sky_cam = tf.placeholder(tf.float32, [None, 12])
    #x_sky_cam_path = tf.placeholder(tf.str, [None, n_step, 1])
    y_ = tf.placeholder(tf.float32, [None, n_target])
    keep_prob = tf.placeholder(tf.float32)

    reader = Reader(config)

    model = Model([x_sky_cam], y_, keep_prob, config)
    prediction = model.prediction
    loss = model.loss
    optimize = model.optimize

    #new a saver to save the model
    saver = tf.train.Saver()

    validation_last_loss = float('inf')

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=60)) as sess:
    #with tf.Session() as sess:

        # initialize all variables
        tf.initialize_all_variables().run()

        for i in range(epoch_size):
            print 'Epoch:', i

            if i%config.test_step == 0:
                print 'Testing...'
                sky_cam_test_input, test_target = reader.get_test_set(test_num)
                test_feed = {x_sky_cam: sky_cam_test_input, keep_prob:1.0}
                test_result = sess.run(prediction, feed_dict=test_feed)
                #'connected'
                #calculate the mse and mae
                mse, mae = MSE_And_MAE(test_target, test_result)
                print "Test MSE: ", mse
                print "Test MAE: ", mae

                sky_cam_train_input, train_target = reader.next_batch()
                train_feed = {x_sky_cam: sky_cam_train_input, keep_prob:0.5}
                train_result = sess.run(prediction, feed_dict=train_feed)
                mse, mae = MSE_And_MAE(train_target, train_result)
                print "Train MSE: ", mse
                print "Train MAE: ", mae


            print 'Training...'
            batch = reader.next_batch()
            train_feed = {x_sky_cam: batch[0], y_:batch[1], keep_prob:0.5}
            sess.run(optimize, feed_dict=train_feed)


            if i%config.print_step == 0:
                print "train loss:",sess.run(loss, feed_dict=train_feed)
                print "validation loss: ", validation_last_loss

            #validation
            print 'Validating...'
            validation_set = reader.get_validation_set()
            validation_feed = {x_sky_cam: validation_set[0], y_:validation_set[1], keep_prob:0.5}
            validation_loss = sess.run(loss,feed_dict=validation_feed)

            #compare the validation with the last loss
            if(validation_loss < validation_last_loss):
                validation_last_loss = validation_loss
            # else:
            #     # break
            #     print "break"

            # print "validation loss: ", validation_loss

if __name__ == "__main__":
    #tf.app.run()
    main()
