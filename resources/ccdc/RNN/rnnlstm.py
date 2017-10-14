#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pandas as pd

#import data
train_data = pd.read_excel('train_data.xlsx', sheetname=0)
train_label = pd.read_excel('train_label.xlsx', sheetname=0)
test_data = pd.read_excel('test_data.xlsx', sheetname=0)
test_label = pd.read_excel('test_label.xlsx', sheetname=0)
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

# 数据归一化
scaler = preprocessing.StandardScaler().fit(train_data)
X_train_minmax = scaler.transform(train_data)
scaler = preprocessing.StandardScaler().fit(test_data)
X_test_minmax = scaler.transform(test_data)

X_train = X_train_minmax
Y_train = train_label.astype(np.float32)
X_test = X_test_minmax
Y_test = test_label.astype(np.float32)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 100

n_inputs = 24   # data input
n_steps = 1    # time steps
n_hidden_units = 100   # neurons in hidden layer
n_classes = 8      # Fault classes (0-7 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


# Define weights
weights = {
    # (24 ,100)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (100 ,8)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (100, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (8, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def GRU(X, weights, biases):
    # hidden layer for input to cell
    # transpose the inputs shape from
    # X ==> (100 batch * 1 steps, 24inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # into hidden
    # X_in = (28 batch * 1 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (20 batch, 1 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    # basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.GRU(n_hidden_units, forget_bias=1.0)
    # lstm cell is divided into two parts (c_state, h_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    # hidden layer for output as the final results
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


pred = GRU(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for step in range(50000):
        # Loop over all batches
        for j in range(8):
            batch_xs = X_train[:(j + 1) * 100]
            batch_xs = batch_xs.reshape([-1, n_steps, n_inputs])
            batch_ys = Y_train[:(j + 1) * 100]
            feed_dict = {x: batch_xs, y: batch_ys}
            _, f = sess.run([train_op, cost], feed_dict=feed_dict)
        if step % 100 == 0:
            print("Step:", '%04d' % (step + 1),
                  "cost=", "{:.9f}".format(f))
            # print(sess.run(accuracy, feed_dict={x: batch_xs,y: batch_ys,})
            Test_data = X_test.reshape((-1, n_steps, n_input))
            Test_label = Y_test
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: Test_data, y: Test_label}))
