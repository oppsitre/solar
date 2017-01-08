# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '12/10/2016'

import tensorflow as tf
#from cnn import cnn_model
class Model:
    """
    This is an multi-modal (three modality) neutral network model used as point/probabilistic forecast
        model digram:
         ---------
        |  lstm   |--------- |
         ---------           |
                             |
         ---------           |           ----------           ------------
        |  lstm   |--------- |----------|   lstm   | --------| regression |
         ---------           |           ----------           ------------
                             |
         ---------           |
        | CNN-lstm |---------|
         ---------

        The model focuses on the multi-modality and this model contain three modalities each of which is lstm, lstm and CNN.
        e.g. This model used to predict solar irradiance and the first and second modalities are the irradiance and meteorological data
            and the third modality is an image dataset and use an CNN to extract the feature. And then concatenating all features into a
            feature as the input of the lstm of the second level. Then we use the output of the last cell as the regressor input to predict
            the value.
        regressor: there are lots of regressors can be used for different purpose. (specific in the config in )
            e.g. linear regression, a fully connected NN with linear regression, support vector regression,
                multi-support vector regression (considering the time dependency)
                quantile regression (used as probabilistic regression)
    """
    def __init__(self, data, target, keep_prob, config):
        """
        @brief The constructor of the model
        @param data: the input the data of the model (features) data[0], data[1], ... for multi-modality
               target: the groundtruth of the model
               keep_prob: use dropout to avoid overfitting (https://www.cs.toronto.edu/%7Ehinton/absps/JMLRdropout.pdf)
               config: the configuration of the model and it may contains following values:
                    n_first_hidden, n_second_hidden, n_hidden_level2, n_fully_connect_hidden, n_target: the params of the network
                    lr: learning rate
                    epsilon, C: params for epsilon-insensitive multi-support regression
        """
        #load the config
        #the input data
        self.data = data
        #self.files = files
        self.target = target
        self.keep_prob = keep_prob
        self.image_data = data[0]
        self.cnn_feat_size = config.cnn_feat_size

        #the network parameters
        self.n_third_hidden = config.n_third_hidden
        self.n_hidden_level2 = config.n_hidden_level2
        self.n_fully_connect_hidden = config.n_fully_connect_hidden
        self.n_target = config.n_target
        self.n_step = config.n_step

        #train params
        self.lr = config.lr

        #loss params (svr params)
        self.epsilon = config.epsilon
        self.C = config.C

        self._prediction = None
        self._optimize = None
        self._loss = None

        self.input_width = config.width
        self.input_height = config.heigth
        self.output_size = config.cnn_feat_size

    @property
    def prediction(self):
        """
        Build the graph of the model and return the prediction value of the model
        NOTE: You can easily treat it as an member of this class (model.prediction to refer)
        """
        print '..................', self.data[0].get_shape()

        if self._prediction is None:
            # build the graph
            # #outputs: [batch_size, n_step, n_hidden] -->> [n_step, batch_size, n_hidden]
            # #output: [batch_size, n_hidden]
            # #get the last output of the lstm as the input of the regressor
            # outputs = tf.transpose(outputs, [1, 0, 2])
            # output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # regression
            w_fc1 = tf.Variable(tf.truncated_normal([12, self.n_fully_connect_hidden]), dtype=tf.float32)
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.n_fully_connect_hidden]), dtype=tf.float32)
            print type(self.data)
            h_fc1 = tf.nn.relu(tf.matmul(self.image_data, w_fc1) + b_fc1)

            # h_fc_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            #multi-support vector regresiion
            self.weight = tf.Variable(tf.truncated_normal([self.n_fully_connect_hidden, self.n_target],stddev=5.0), dtype=tf.float32)
            bias = tf.Variable(tf.constant(0.1, shape=[self.n_target]), dtype=tf.float32)

            self._prediction = tf.matmul(h_fc1, self.weight) + bias

        return self._prediction

    @property
    def loss(self):
        """
        Define the loss of Fthe model and you can modify this section by using different regressor
        """
        if self._loss is None:
            #compute the ||w||2
            #use the w^T * W to compute and the sum the diag to get the result
            m = tf.matmul(tf.transpose(self.weight,[1,0]), self.weight)
            diag = tf.matrix_diag_part(m)
            w_sqrt_sum = tf.reduce_sum(diag)

            #the loss of the trian set
            diff = self.prediction - self.target
            err = tf.sqrt(tf.reduce_sum(tf.square(diff), reduction_indices=1)) - self.epsilon
            err_greater_than_espilon = tf.cast(err > 0, tf.float32)
            total_err = tf.reduce_sum(tf.mul(tf.square(err), err_greater_than_espilon))

            self._loss = 0.5 * w_sqrt_sum + self.C * total_err
            # self._loss = total_err
        return self._loss


    @property
    def optimize(self):
        """
        Define the optimizer of the model used to train the model
        """
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize
