import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
from pylab import *
from tsne import tsne
from Batcher import Batcher
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import layers

class Autoencoder(object):

    """Docstring for Autoencoder. """

    def __init__(self, shape=[100, 40], classifier=[15], max_iter=50000):
        """TODO: to be defined1. """
        self.shape = shape
        self.max_iter = max_iter
        self.classifier = classifier

    def fit(self, X, Y):
        """TODO: Docstring for fit.
        :returns: TODO

        """
        pl_x = X.shape[1]
        pl_y = Y.shape[1]

        x = tf.placeholder(tf.float32, [None, pl_x])
        y = tf.placeholder(tf.float32, [None, pl_y])

        encoder = layers.stack(x, layers.fully_connected, self.shape)
        decoder = layers.stack(encoder, layers.fully_connected, list(reversed(self.shape))[1:] + [pl_x])
        classifier = layers.stack(encoder, layers.fully_connected, self.classifier + [pl_y])

        loss_encoder = tf.losses.mean_pairwise_squared_error(x, decoder)
        loss_class = tf.losses.mean_pairwise_squared_error(y, classifier)

        opt_enc = tf.train.RMSPropOptimizer(0.002, momentum=0.1).minimize(loss_encoder)
        opt_cls = tf.train.RMSPropOptimizer(0.05).minimize(loss_class)

        batcher = Batcher(X, Y)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        try:
            for i in range(self.max_iter):
                xs, ys = batcher.next_batch(4)
                feed_dict = {x: xs, y: ys}
                loss, _ = self.sess.run([loss_encoder, opt_enc], feed_dict=feed_dict)
                print('encoding loss:', 1000*loss, end='\r')
        except KeyboardInterrupt:
            print('interrupted!')
        return self.sess.run([encoder, decoder], feed_dict={x:X, y:Y})


def main():
    data_path = '/home/charlesxu/Workspace/xcc/projects/ccdc/essay_ccdc/code/asort/1202/Auto+softmax'

    data_test_orig = pd.read_excel(
        os.path.join(data_path, 'train_data.xlsx')).values

    data_label = pd.read_excel(
        os.path.join(data_path, 'train_label.xlsx')).values

    scaler = MinMaxScaler([0.1, 2])
    data_test = scaler.fit_transform(data_test_orig)


    model = Autoencoder(shape=[150, 8])
    data_embed, decoded = model.fit(data_test, data_label)

    enc_label1 = zeros([data_test.shape[0], 2])
    enc_label2 = zeros([data_test.shape[0], 2])
    enc_label1[:, 1] = 1
    enc_label2[:, 0] = 1
    enc_label = vstack([enc_label1, enc_label2])
    enc_data = vstack([data_test, decoded])
    tsne(enc_data, enc_label, title='decoded-orig', unbalanced=False)
    tsne(decoded, data_label, title='decoded')
    tsne(data_test, data_label, title='orig')
    tsne(data_embed, data_label, title='encoded')

    show()


if __name__ == "__main__":
    main()


