from typing import Dict

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow_ranking.python import utils

from toyml.match.network import TwoTowerNetwork


class YouTubeDNNNetwork(TwoTowerNetwork):
    """YouTube DNN network for match"""

    def __init__(self,
                 context_feature_columns=None,
                 example_feature_columns=None,
                 sparse_features: Dict = None,
                 sequence_features: Dict = None,
                 hidden_layer_dims=None,
                 activation=tf.nn.relu,
                 name='youtube_dnn_network',
                 **kwargs):
        if not example_feature_columns or not hidden_layer_dims:
            raise ValueError('example_feature_columns or hidden_layer_dims must not be empty.')
        super(YouTubeDNNNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            name=name,
            **kwargs)
        self._hidden_layer_dims = [int(d) for d in hidden_layer_dims]
        self._activation = activation

        layers = []
        for _, layer_width in enumerate(self._hidden_layer_dims):
            layers.append(Dense(units=layer_width))
            layers.append(tf.keras.layers.Activation(activation=self._activation))
        self._scoring_layers = layers

    def score(self,
              context_inputs=None,
              example_inputs=None,
              mask=None,
              training=None):
        context_output = context_inputs
        for layer in self._scoring_layers:
            context_output = layer(context_output, training=training)
        outputs = tf.einsum('bf,blf -> bl', context_output, example_inputs)
        scores = Softmax()(outputs)
        return scores

    def get_config(self):
        config = super(YouTubeDNNNetwork, self).get_config()
        config.update({
            'hidden_layer_dims': self._hidden_layer_dims,
            'activation': self._activation
        })
        return config


def cross_entropy_loss(labels, logits):
    """Computes the negative log likehood loss ."""
    labels = tf.compat.v1.where(utils.is_label_valid(labels), labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(utils.is_label_valid(labels), logits, tf.zeros_like(logits))

    epsilon_ = constant_op.constant(K.epsilon(), dtype=logits.dtype.base_dtype)
    output = clip_ops.clip_by_value(logits, epsilon_, 1. - epsilon_)

    bce = labels * math_ops.log(output + K.epsilon())
    bce += (1 - labels) * math_ops.log(1 - output + K.epsilon())
    return tf.compat.v1.losses.compute_weighted_loss(losses=-bce)
