import tensorflow as tf
import numpy as np
from sklearn import preprocessing
# import seaborn as sns
import pandas as pd
from pylab import *

data = pd.read_excel('1202/Auto+softmax/train_data.xlsx')
label = pd.read_excel('1202/Auto+softmax/train_label.xlsx')
# data = pd.concat([label, data], axis=1)


def get_index(label, n):
    return list(label.iloc[n]).index(1)


tmp = list()
for i in range(len(label)):
    tmp.append(get_index(label, i))
labels = set(tmp)
tmp = pd.DataFrame(tmp)

plots = list()
for i in labels:
    plots.append(data.loc[tmp[tmp[0] == i].index])


def get_shape(a, b, c, data=plots):
    cr = list('rgb')
    scatter(plots[c].values[:, a], plots[c].values[:, b], c=cr[c % len(cr)])

# for i in range(8):
#     subplot(2, 4, i)
#     get_shape(0, 1, 2)


xr = data
yr = label
X_train = np.array(xr)
Y_train = np.array(yr)
print(X_train.shape)
print(Y_train.shape)

scaler1 = preprocessing.StandardScaler()
scaler1.fit(X_train)
X = scaler1.transform(X_train)

scaler2 = preprocessing.MinMaxScaler()
scaler2.fit(X)
X = scaler2.transform(X)

ttt = .75
X = X * ttt + (1 - ttt) / 2

Y = Y_train.astype(np.float32)


def ae(struct_of_net=[150, 8]):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 24])
    W1 = tf.Variable(tf.zeros([24, struct_of_net[0]]))
    b1 = tf.Variable(tf.zeros([struct_of_net[0]]))
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    W3 = tf.Variable(tf.random_normal([struct_of_net[0], struct_of_net[-1]]))
    b3 = tf.Variable(tf.random_normal([struct_of_net[-1]]))
    y3 = tf.nn.sigmoid(tf.matmul(y1, W3) + b3)

    W4 = tf.Variable(tf.random_normal([struct_of_net[-1], struct_of_net[0]]))
    b4 = tf.Variable(tf.random_normal([struct_of_net[0]]))
    y4 = tf.nn.sigmoid(tf.matmul(y3, W4) + b4)

    W2 = tf.Variable(tf.zeros([struct_of_net[0], 24]))
    b2 = tf.Variable(tf.zeros([24]))
    y2 = tf.nn.sigmoid(tf.matmul(y4, W2) + b2)
    y_true = x
    # Define cost and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y2, 2))
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1000):
            # Loop over all batches
            for i in range(10):
                fd = {x: X[:(i + 1) * 100, :]}
                _, c = sess.run([optimizer, cost], feed_dict=fd)
            # Display logs per epoch step
            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))
        print("AutoEncoder Optimization Finished!")
        datas, feature = sess.run([y2, y3], feed_dict={x: X})
        figure()
        scatter(datas[:, 0], datas[:, 1], c='r')
        scatter(X[:, 0], X[:, 1], c='g')
        xlabel(str(struct_of_net))
        show()
    return [datas, X], feature
