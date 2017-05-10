# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import xlrd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import os

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

# Import data

sess = tf.InteractiveSession()

def xlsread(xls):
    fname = xls
    bk = xlrd.open_workbook(fname)
    try:
        sh = bk.sheet_by_name("Sheet1")
    except:
        print('no sheet in %s named Sheet1' % fname)
    nrows = sh.nrows
    ncols = sh.ncols
    print('(nrows %d, ncols %d)' % (nrows, ncols))
    row_list = []
    for i in range(0, nrows):
        row_data = sh.row_values(i)
        row_list.append(row_data)
    return row_list


x_train = xlsread('1202/Auto+softmax/train_data.xlsx')
y_train = xlsread('1202/Auto+softmax/train_label.xlsx')
x_test = xlsread('1202/Auto+softmax/test_data.xlsx')
y_test = xlsread('1202/Auto+softmax/test_label.xlsx')
X_train = np.array(x_train)
Y_train = np.array(y_train)
X_test = np.array(x_test)
Y_test = np.array(y_test)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# 归一化处理成0-1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_minmax = scaler.transform(X_train)
scaler = preprocessing.StandardScaler().fit(X_test)
X_test_minmax = scaler.transform(X_test)

X_TRAIN = X_train_minmax
Y_TRAIN = Y_train.astype(np.float32)
X_TEST = X_test_minmax
Y_TEST = Y_test.astype(np.float32)

ttt = .75
X_TRAIN = X_TRAIN * ttt + (1 - ttt) / 2
X_TEST = X_TEST * ttt + (1 - ttt) / 2


def autoencoder_acc(hidden):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 24])
    y_ = tf.placeholder(tf.float32, [None, 8])

    W1 = tf.Variable(tf.random_normal([24, hidden]))
    b1 = tf.Variable(tf.zeros([hidden]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([hidden, 8]))
    b2 = tf.Variable(tf.zeros([8]))
    y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

    W3 = W1 = tf.Variable(tf.random_normal([8, hidden]))
    b3 = tf.Variable(tf.zeros([hidden]))
    y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

    W4 = tf.Variable(tf.random_normal([hidden, 24]))
    b4 = tf.Variable(tf.zeros([24]))
    y4 = tf.nn.sigmoid(tf.matmul(y3, W4) + b4)
    y_true = x
    # Define cost and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y4, 2))
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    sess = tf.InteractiveSession()
    sess.run(init)
    for epoch in range(1000):
        # Loop over all batches
        for i in range(9):
            fd = {x: X_TRAIN[:(i + 1) * 100, :]}
            _, c = sess.run([optimizer, cost], feed_dict=fd)
        # Display logs per epoch step
        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c), end='   \r')
    print("AutoEncoder Optimization Finished!")

    # build NN model
    W5 = tf.Variable(tf.random_normal([8, 8]))
    b5 = tf.Variable(tf.zeros([8]))
    y_pred = tf.nn.softmax(tf.matmul(y2, W5) + b5)

    cross_entropy = - \
        tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_pred, 1e-6, 1.0)))

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    # saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(init)
    accu = 0
    ff = [1e+9] * 2
    error = []
    precision = []
    for step in range(7000):
        # Loop over all batches
        for j in range(9):
            feed_dict = {x: X_TRAIN[:(j + 1) * 100], y_: Y_TRAIN[:(j + 1) * 100]}
            _, f = sess.run([train_step, cross_entropy], feed_dict=feed_dict)
        # Display logs per epoch step
        if step % 100 == 0:
            g = sess.run(accuracy, feed_dict={x: X_TEST, y_: Y_TEST})
            print("accuracy_rate=", "%3f" % g, end='   \r')
            if g >= accu:
                accu = g

            ff.append(f)
            error.append(f)
            precision.append(g)
            ff.pop(0)
            if (ff[0] - ff[1]) < -10:
                break
            if f < 10 * .0000000001:
                break
    y_p = sess.run(y_pred, feed_dict={x: X_TEST, y_: Y_TEST})
    y_p = np.round(y_p)

    with open('autoencoder.csv', 'a') as f:
        print('%d, %f' % (hidden, accu), file=f)
    print("Optimization Finished!")
    return accu


def mlp_acc(hidden):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 24])
    y_ = tf.placeholder(tf.float32, [None, 8])

    W1 = tf.Variable(tf.random_normal([24, hidden]))
    b1 = tf.Variable(tf.zeros([hidden]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([hidden, 8]))
    b2 = tf.Variable(tf.zeros([8]))
    y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

    # build NN model
    W5 = tf.Variable(tf.random_normal([8, 8]))
    b5 = tf.Variable(tf.zeros([8]))
    y_pred = tf.nn.softmax(tf.matmul(y2, W5) + b5)

    cross_entropy = - \
        tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_pred, 1e-6, 1.0)))

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    # saver = tf.train.Saver()
    sess.run(init)
    accu = 0
    ff = [1e+9] * 2
    error = []
    precision = []
    for step in range(7000):
        # Loop over all batches
        for j in range(9):
            feed_dict = {x: X_TRAIN[:(j + 1) * 100], y_: Y_TRAIN[:(j + 1) * 100]}
            _, f = sess.run([train_step, cross_entropy], feed_dict=feed_dict)
        # Display logs per epoch step
        if step % 100 == 0:
            g = sess.run(accuracy, feed_dict={x: X_TEST, y_: Y_TEST})
            print("accuracy_rate=", "%3f" % g, end='    \r')
            if g >= accu:
                accu = g

            ff.append(f)
            error.append(f)
            precision.append(g)
            ff.pop(0)
            if (ff[0] - ff[1]) < -10:
                break
            if f < 10 * .0000000001:
                break
    print(accu)
    y_p = sess.run(y_pred, feed_dict={x: X_TEST, y_: Y_TEST})
    y_p = np.round(y_p)

    with open('mlp.csv', 'a') as f:
        print('%d, %f' % (hidden, accu), file=f)
    print("Optimization Finished!")
    return accu


def acc():
    # hiddens = list(range(8, 24, 3))
    # hiddens.extend(range(24, 50, 5))
    # hiddens.extend(range(50, 200, 10))
    # hiddens.extend(range(200, 300, 40))
    hiddens = list(range(300, 500, 10))
    hiddens.extend(range(500, 900, 50))
    hiddens.extend(range(900, 9000, 50))

    f = open('accuracy.csv', 'w')
    print('%s, %s, %s' % ('hidden layer', 'mlp', 'autoencoder'), file=f)
    i=0
    for hidden in hiddens:
        with tf.device("/gpu:{}".format(i%4)):
            acc_mlp = mlp_acc(hidden)
            acc_ae = autoencoder_acc(hidden)
        print('%d, %f, %f' % (hidden, acc_ae, acc_mlp), file=f)
        i+=1
    f.close()


def main():
    acc()


if __name__ == '__main__':
    main()
