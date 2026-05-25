import os
import shutil

import numpy as np
import tensorflow as tf


class LogisticalRegression:
    def __init__(self, lr=0.01, reg_param=0.01):
        self.lr = lr
        self.reg_param = reg_param
        self.model = None

    def fit(self,
            X_train,
            y_train,
            model_path,
            batch_size=128,
            epochs=10):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],), name='x'),
            tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                name='y_pred',
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_param)),
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        self.model.save(model_path)

    @classmethod
    def predict(cls,
                X,
                model_path,
                batch_size=128):
        model = tf.keras.models.load_model(model_path)
        return model.predict(X, batch_size=batch_size, verbose=0).ravel()
