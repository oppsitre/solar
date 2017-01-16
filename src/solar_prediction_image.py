# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import tensorflow as tf
import numpy as np
import json
from collections import namedtuple
from reader_image import Reader
from model_cnn import Model
from util import MSE_And_MAE, test_figure_plot
import sys
import datetime
import time
t = time.strftime('%m%d%H%M',time.localtime(time.time()))
#log_file = open('cnn_log_'+ t,'w')
#sys.stdout = log_file
def main(_):
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
    x_sky_cam = tf.placeholder(tf.float32, [None, n_step, height_image, width_image])
    #x_sky_cam_path = tf.placeholder(tf.str, [None, n_step, 1])
    y_ = tf.placeholder(tf.float32, [None, n_target])
    keep_prob = tf.placeholder(tf.float32)

    reader = Reader(config)

    model = Model([x_sky_cam], y_, keep_prob, config)
    prediction = model.prediction
    loss = model.loss
    optimize = model.optimize
    init_op = tf.global_variables_initializer()
    #new a saver to save the model
    saver = tf.train.Saver()

    validation_last_loss = float('inf')
    tensor_config = tf.ConfigProto()
    tensor_config.gpu_options.allow_growth = True
    tensor_config.log_device_placement = True
    with tf.Session(config=tensor_config) as sess:
        # initialize all variables
        sess.run(init_op)
        save_path = './cnn_model.ckpt'
        #saver.restore(sess, save_path)
        for i in range(epoch_size):
            print 'Epoch:', i

            if i%config.test_step == 0:
                sky_cam_test_input, test_target = reader.get_test_set(test_num)
                test_feed = {x_sky_cam: sky_cam_test_input, keep_prob:1.0}
                test_result = sess.run(prediction, feed_dict=test_feed)
                #'connected'
                #calculate the mse and mae
                mse, mae = MSE_And_MAE(test_target, test_result)
                print "Test MSE: ", mse
                print "Test MAE: ", mae

                sky_cam_train_input, train_target = reader.next_batch()
                train_feed = {x_sky_cam: sky_cam_train_input, keep_prob:1.0}
                train_result = sess.run(prediction, feed_dict=train_feed)
                mse, mae = MSE_And_MAE(train_target, train_result)
                print "Train MSE: ", mse
                print "Train MAE: ", mae



            batch = reader.next_batch()
            train_feed = {x_sky_cam: batch[0], y_:batch[1], keep_prob:0.5}
            sess.run(optimize, feed_dict=train_feed)


            if i%config.print_step == 0:
                print "train loss:",sess.run(loss, feed_dict=train_feed)
                print "validation loss: ", validation_last_loss

            #validation
            validation_set = reader.get_validation_set()
            validation_feed = {x_sky_cam: validation_set[0], y_:validation_set[1], keep_prob:0.5}
            validation_loss = sess.run(loss,feed_dict=validation_feed)

            #compare the validation with the last loss
            if(validation_loss < validation_last_loss):
                validation_last_loss = validation_loss

            sp = saver.save(sess, save_path)
            print("Model saved in file: %s" % sp)
            # else:
            #     # break
            #     print "break"

            # print "validation loss: ", validation_loss

if __name__ == "__main__":
    tf.app.run()
