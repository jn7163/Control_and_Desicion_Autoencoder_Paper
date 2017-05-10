# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import xlrd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

# Import data


def xlsread(xls):
    fname = xls
    bk = xlrd.open_workbook(fname)
    shxrange = range(bk.nsheets)
    try:
        sh = bk.sheet_by_name("Sheet1")
    except:
        print('no sheet in %s named Sheet1' % fname)
    nrows = sh.nrows
    ncols = sh.ncols
    print('(nrows %d, ncols %d)' % (nrows, ncols))
    cell_value = sh.cell_value(1, 1)
    row_list = []
    for i in range(0, nrows):
        row_data = sh.row_values(i)
        row_list.append(row_data)
    return row_list

x_train = xlsread('train_data.xlsx')
y_train = xlsread('train_label.xlsx')
x_test = xlsread('test_data.xlsx')
y_test = xlsread('test_label.xlsx')
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

# Create the model
x = tf.placeholder(tf.float32, [None, 24])
y_ = tf.placeholder(tf.float32, [None, 8])

W1 = tf.Variable(tf.random_normal([24, 150]))
b1 = tf.Variable(tf.zeros([150]))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.random_normal([150, 8]))
b2 = tf.Variable(tf.zeros([8]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

W3 = W1 = tf.Variable(tf.random_normal([8, 150]))
b3 = tf.Variable(tf.zeros([150]))
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

W4 = tf.Variable(tf.random_normal([150, 24]))
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
              "cost=", "{:.9f}".format(c))
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
saver = tf.train.Saver()
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
        print("Step:", '%04d' % (step + 1),
              "cross_entropy=", "{:.9f}".format(f))
        #print(sess.run(accuracy, feed_dict={x: X, y_: Y}))
        g = sess.run(accuracy, feed_dict={x: X_TEST, y_: Y_TEST})
        print("accuracy_rate=", "%3f" % g)
        if g >= accu:
            accu = g
            saver.save(sess, 'ckpt/epoch_d%d_model_accuracy_%3f' % (step, g))
            print('saved')

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

print("Optimization Finished!")


# 误差曲线&误辨率
[row, column] = Y_TEST.shape
# 定义类别存放向量
class_output0 = np.zeros(row)
net_output0 = np.zeros(row)
print(y_p)
# 将类别变量转换为十进制数
for i in range(row):
    for j in range(column):
        if Y_TEST[i, j] == 1:
            class_output0[i] = j
            break
for i in range(row):
    for j in range(column):
        if y_p[i, j] == 1:
            net_output0[i] = j
            break

num1 = []
for i in range(8):
    counter = 0
    for j in range(row):
        if class_output0[j] == i:
            counter = counter + 1
    num1.append(counter)
print(num1)

num2 = []
for i in range(8):
    counter = 0
    for j in range(row):
        if net_output0[j] == i:
            counter = counter + 1
    num2.append(counter)
print(num2)


# 绘制误差曲线
t = np.arange(100)
fig = plt.figure(num=1, figsize=(10, 10))
ax1 = fig.add_subplot(1, 2, 1)
l0 = ax1.plot(t, class_output0, color='red', linewidth=1.0, marker='o')
plt.xlim((-1, 100))
plt.ylim((0, 9))
plt.xlabel('samples data')
plt.ylabel('fault diagnosis classifications')
ax1.set_title('original curve graph')

ax2 = fig.add_subplot(1, 2, 2)
l3 = ax2.plot(t, net_output0, color='b', linewidth=1.0, marker='*')
plt.xlim((-1, 100))
plt.ylim((0, 9))
plt.xlabel('samples data')
plt.ylabel('fault diagnosis classifications')
ax2.set_title('recognition curve graph')


fig = plt.figure(num=2, figsize=(10, 10))
plt.plot(t, class_output0, label='actual fault classification', color='m', linewidth=2.0, marker='*')
plt.plot(t, net_output0, label='recognintion fault classification', color='g', linewidth=1.0, marker='o')
plt.xlabel('samples data')
plt.ylabel('fault classification')
plt.xlim((-1, 100))
plt.ylim((0, 9))
plt.title('recognition data classification curve graph')
plt.legend()

# 误差变化曲线
s = np.arange(len(error))
fig = plt.figure(num=3, figsize=(10, 10))
ax4 = fig.add_subplot(1, 1, 1)
plt.plot(s, error, 'c-*')
plt.xlim((-1, len(error)))
plt.ylim((100, 3000))
plt.xlabel('steps')
plt.ylabel('cross entropy')
ax4.set_title('error curve graph')


# 准确率曲线
r = np.arange(len(precision))
fig = plt.figure(num=4, figsize=(10, 10))
ax5 = fig.add_subplot(1, 1, 1)
plt.plot(r, precision, 'k-o')
plt.xlim((-1, len(precision)))
plt.ylim((0, 1))
plt.xlabel('steps')
plt.ylabel('precision rate')
ax5.set_title('precision rate curve graph')


plt.show()
