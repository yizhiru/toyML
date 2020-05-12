import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class LogisticalRegression:
    def __init__(self, lr=0.01, reg_param=0.01):
        self.lr = lr
        self.reg_param = reg_param

    def fit(self,
            X_train,
            y_train,
            model_path,
            batch_size=128,
            epochs=10):
        num_labels = 2
        input_dim = X_train.shape[1]
        y_train = tf.keras.utils.to_categorical(y_train)

        # tf Graph Input
        x = tf.placeholder(tf.float32, shape=(None, input_dim), name='x')
        y = tf.placeholder(tf.float32, shape=(None, num_labels), name='y')
        # model params
        weights = tf.Variable(tf.truncated_normal(shape=[input_dim, num_labels]), name='weights')
        biases = tf.Variable(tf.zeros([num_labels]), name='biases')

        # computation
        logits = tf.matmul(x, weights) + biases
        y_pred = tf.nn.softmax(logits, name='y_pred')
        # loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)) + self.reg_param * tf.nn.l2_loss(
            weights)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                avg_loss = 0.
                steps_per_epoch = (X_train.shape[0] + batch_size - 1) // batch_size
                for i in range(steps_per_epoch):
                    batch_x = X_train[i * batch_size: (i + 1) * batch_size]
                    batch_y = y_train[i * batch_size: (i + 1) * batch_size]
                    _, c = sess.run([optimizer, loss],
                                    feed_dict={x: batch_x, y: batch_y})
                    avg_loss += c / steps_per_epoch
                print(f'epoch: {epoch}/{epochs}, avg loss={avg_loss:.4f}')
            print('train finished.')

            # save model
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            tf.saved_model.simple_save(sess,
                                       model_path,
                                       inputs={'x': x},
                                       outputs={'y_pred': y_pred})

    @classmethod
    def predict(cls,
                X,
                model_path,
                batch_size=128):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)
            steps = (X.shape[0] + batch_size - 1) // batch_size
            y_pred = []
            for i in range(steps):
                batch_x = X[i * batch_size: (i + 1) * batch_size]
                batch_y_pred = sess.run('y_pred:0', feed_dict={'x:0': batch_x})
                y_pred.append(batch_y_pred[:, 1])
            return np.concatenate(y_pred).ravel()
