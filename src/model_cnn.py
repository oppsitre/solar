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
        self.fc_1 = 4096
        self.fc_2 = 1024

    def weight_varible(self, shape):
        # initial = tf.truncated_normal(shape, stddev=0.1)
        weights = tf.get_variable('weights', shape, initializer=tf.random_normal_initializer(0, stddev=5.0))
        return weights

    def bias_variable(self, shape):
        # initial = tf.constant(0.1, shape=shape)
        bias = tf.get_variable('bias', shape, initializer=tf.constant_initializer(0.1))
        return bias

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def cnn_model(self, x, keep_prob):
        x_image = tf.reshape(x, [-1, self.input_height, self.input_width, 1])
        with tf.variable_scope('conv_1') as scope:
            W_conv1 = self.weight_varible([3, 3, 1, 32])
            b_conv1 = self.bias_variable([32])

            # conv layer-1
            # x = tf.placeholder(tf.float32, [None, 784])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)
            #h_pool1 = h_conv1

            axis = list(range(len(h_pool1.get_shape()) - 1))
            mean, variance = tf.nn.moments(h_pool1, axis)
            #print mean.get_shape(), variance.get_shape()
            batch_normal = tf.nn.batch_normalization(h_pool1, mean, variance, None, None, 0.001)
            #return h_pool1

        with tf.variable_scope('conv_2') as scope:
            # conv layer-2
            W_conv2 = self.weight_varible([3, 3, 32, 64])
            b_conv2 = self.bias_variable([64])

            h_conv2 = tf.nn.relu(self.conv2d(batch_normal, W_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)
            #h_pool2 = h_conv2


            axis = list(range(len(h_pool2.get_shape()) - 1))
            mean, variance = tf.nn.moments(h_pool2, axis)
            #print mean.get_shape(), variance.get_shape()

            batch_normal = tf.nn.batch_normalization(h_pool2, mean, variance, None, None, 0.001)

        with tf.variable_scope('conv_3') as scope:
            # conv layer-2
            W_conv3 = self.weight_varible([3, 3, 64, 64])
            b_conv3 = self.bias_variable([64])

            h_conv3 = tf.nn.relu(self.conv2d(batch_normal, W_conv3) + b_conv3)
            h_pool3 = self.max_pool_2x2(h_conv3)
            #h_pool2 = h_conv2

            #Batch_Normalization
            axis = list(range(len(h_pool2.get_shape()) - 1))
            mean, variance = tf.nn.moments(h_pool3, axis)
            batch_normal = tf.nn.batch_normalization(h_pool3, mean, variance, None, None, 0.001)

        with tf.variable_scope('fc_1') as scope:
            # full connection
            s = batch_normal.get_shape().as_list()
            #print 'X1,X2,X3,X4', s
            W_fc1 = self.weight_varible([s[1]*s[2]*s[3], self.fc_1])
            b_fc1 = self.bias_variable([self.fc_1])

            batch_normal_flat = tf.reshape(batch_normal, [-1, s[1]*s[2]*s[3]])
            h_fc1 = tf.nn.relu(tf.matmul(batch_normal_flat, W_fc1) + b_fc1)

            #Batch_Normalization
            axis = list(range(len(h_fc1.get_shape()) - 1))
            mean, variance = tf.nn.moments(h_fc1, axis)
            batch_normal = tf.nn.batch_normalization(h_fc1, mean, variance, None, None, 0.001)

            # dropout
            h_fc1_drop = tf.nn.dropout(batch_normal, keep_prob)
            #print 'h_fc1_drop:', h_fc1_drop.get_shape()

            #return h_fc1_drop
        with tf.variable_scope('fc_2') as scope:
            # full connection
            #s = batch_normal.get_shape().as_list()
            #print 'X1,X2,X3,X4', s
            W_fc2 = self.weight_varible([(h_fc1_drop.get_shape())[1], self.fc_2])
            b_fc2 = self.bias_variable([self.fc_2])

            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

            #Batch_Normalization
            axis = list(range(len(h_fc2.get_shape()) - 1))
            mean, variance = tf.nn.moments(h_fc2, axis)
            batch_normal = tf.nn.batch_normalization(h_fc2, mean, variance, None, None, 0.001)

            # dropout
            h_fc2_drop = tf.nn.dropout(batch_normal, keep_prob)
            # print W_fc1.name, W_fc1.get_shape()
            # print b_fc1.name, b_fc1.get_shape()
            # print h_pool2_flat.name, h_pool2_flat.get_shape()
            # print h_fc1_drop.name, h_fc1_drop.get_shape()
            return h_fc2_drop
    @property
    def prediction(self):
        """
        Build the graph of the model and return the prediction value of the model
        NOTE: You can easily treat it as an member of this class (model.prediction to refer)
        """
        print '..................', self.data[0].get_shape()

        if self._prediction is None:
            # build the graph
            cnn_out = None
            with tf.variable_scope('image') as scope:
                for i in range(self.n_step):
                    tmp_out = self.cnn_model(self.image_data[:, i, :, :], self.keep_prob)
                    tmp_out = tf.reshape(tmp_out, [-1, 1, self.cnn_feat_size])
                    if cnn_out is None:
                        cnn_out = tmp_out
                    else:
                        cnn_out = tf.concat(1, [tmp_out,cnn_out])
                    scope.reuse_variables()
                    #print '!!!!!', cnn_out.name, cnn_out.get_shape()

                    # h_fc1_drop_2 = cnn_model(x, keep_prob)
                    # h_fc1_drop_2 = h_fc1_drop_2[:,:512]
                    # print h_fc1_drop_2.name
            print cnn_out.name, cnn_out.get_shape()

            with tf.variable_scope("first_level3"):
                cell_3 = tf.nn.rnn_cell.LSTMCell(self.n_third_hidden, state_is_tuple=True)
                outputs_3, state3 = tf.nn.dynamic_rnn(cell_3, cnn_out, dtype=tf.float32)

            #2nd level lstm
            with tf.variable_scope("second_level"):
                cell_level2 = tf.nn.rnn_cell.LSTMCell(self.n_hidden_level2, state_is_tuple=True)
                outputs, state_level2 = tf.nn.dynamic_rnn(cell_level2, outputs_3, dtype=tf.float32)


            #outputs: [batch_size, n_step, n_hidden] -->> [n_step, batch_size, n_hidden]
            #output: [batch_size, n_hidden]
            #get the last output of the lstm as the input of the regressor
            outputs = tf.transpose(outputs, [1, 0, 2])
            output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # regression
            w_fc1 = tf.Variable(tf.truncated_normal([self.n_hidden_level2, self.n_fully_connect_hidden], stddev = 5.0), dtype=tf.float32)
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.n_fully_connect_hidden]), dtype=tf.float32)
            h_fc1 = tf.nn.relu(tf.matmul(output, w_fc1) + b_fc1)

            # h_fc_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            #multi-support vector regresiion
            self.weight = tf.Variable(tf.truncated_normal([self.n_fully_connect_hidden, self.n_target], stddev = 5.0), dtype=tf.float32)
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
