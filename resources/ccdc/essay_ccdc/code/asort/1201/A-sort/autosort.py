# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import xlrd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

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
	print('(nrows %d, ncols %d)' % (nrows,ncols))
	cell_value = sh.cell_value(1,1)
	row_list = []
	for i in range(0,nrows):
		row_data = sh.row_values(i)
		row_list.append(row_data)
	return row_list

xr = xlsread('shuju.xlsx')
yr = xlsread('biaoqian1.xlsx')
X_train = np.array(xr)
Y_train = np.array(yr)
print(X_train.shape)
print(Y_train.shape)
#归一化处理成0-1
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X = X_train_minmax
Y = Y_train

# Create the model
x = tf.placeholder(tf.float32, [None, 24])
y_ = tf.placeholder(tf.float32, [None, 9])
W1 = tf.Variable(tf.zeros([24, 10]))
b1 = tf.Variable(tf.zeros([10]))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
W2 = W1 = tf.Variable(tf.zeros([10, 24]))
b2 = tf.Variable(tf.zeros([24]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
y_true = X

# Define cost and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y2, 2))
optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.InteractiveSession()
sess.run(init)
for epoch in range(60):
    # Loop over all batches
    for i in range(10):
		fd = {x: X[:,:(i+1)*100]}
		_, c = sess.run([optimizer, cost], feed_dict= fd)
    # Display logs per epoch step
    if epoch % 1 == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))
print("AutoEncoder Optimization Finished!")

#build NN model
W3 = tf.Variable(tf.zeros([10, 9]))
b3 = tf.Variable(tf.zeros([9]))
y_pred = tf.nn.softmax(tf.matmul(y1, W3) + b3)

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_*tf.log(y_pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.InteractiveSession()
sess.run(init)

for step in range(1000):
    # Loop over all batches
    for j in range(10):
		_, f = sess.run([train_step,cross_entropy ],feed_dict = {x: X[:,:(j+1)*100],y_: Y[:,:(j+1)*100]})
    # Display logs per epoch step
    if step % 100 == 0:
        print("Step:", '%04d' % (step+1),
              "loss=", "{:.9f}".format(f))																																																																																																																																																																																																								
		#print(sess.run(accraacy,feed_dict = {x:X_train_minmax,y_:Y_train}))
print("Optimization Finished!")
