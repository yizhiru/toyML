from typing import Dict

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow_ranking.python import utils

from toyml.match.network import TwoTowerNetwork
from toyml.utils import build_embedding_layer


class DSSMNetwork(TwoTowerNetwork):
    """DSSM network for match"""

    def __init__(self,
                 context_feature_columns=None,
                 example_feature_columns=None,
                 sparse_features: Dict = None,
                 sequence_features: Dict = None,
                 hidden_layer_dims=None,
                 activation=tf.nn.relu,
                 gamma=1.0,
                 name='youtube_dnn_network',
                 **kwargs):
        if not example_feature_columns or not hidden_layer_dims:
            raise ValueError('example_feature_columns or hidden_layer_dims must not be empty.')
        super(DSSMNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            name=name,
            **kwargs)
        self._hidden_layer_dims = [int(d) for d in hidden_layer_dims]
        self._activation = activation
        self._gamma = gamma

        def _dnn():
            layers = []
            for _, layer_width in enumerate(self._hidden_layer_dims):
                layers.append(Dense(units=layer_width))
                layers.append(tf.keras.layers.Activation(activation=self._activation))
            return layers

        self._user_embed_layers = _dnn()
        self._item_embed_layers = _dnn()

    def score(self,
              context_inputs=None,
              example_inputs=None,
              mask=None,
              training=None):
        batch_size = tf.shape(example_inputs)[0]
        list_size = tf.shape(example_inputs)[1]

        user_embed = context_inputs
        for layer in self._user_embed_layers:
            user_embed = layer(user_embed, training=training)
        item_embed = example_inputs
        for layer in self._item_embed_layers:
            item_embed = layer(item_embed, training=training)

        # expand user embedding to be of [batch_size, list_size, ...]
        user_embed = tf.expand_dims(input=user_embed, axis=1)
        user_embed = tf.gather(user_embed, tf.zeros([list_size], tf.int32), axis=1)
        user_embed = utils.reshape_first_ndims(user_embed, 2, [batch_size, list_size])

        similarities = cosine_similarity(user_embed, item_embed)
        scores = tf.math.scalar_mul(self._gamma, similarities)
        scores = Softmax()(scores)
        return scores

    def get_config(self):
        config = super(DSSMNetwork, self).get_config()
        config.update({
            'hidden_layer_dims': self._hidden_layer_dims,
            'activation': self._activation
        })
        return config


def cosine_similarity(tensor1, tensor2, axis=-1):
    tensor1 = nn.l2_normalize(tensor1, axis=axis)
    tensor2 = nn.l2_normalize(tensor2, axis=axis)
    return math_ops.reduce_sum(tensor1 * tensor2, axis=axis)


def dssm_loss(labels, logits):
    """Computes the DSMM model loss ."""
    labels = tf.compat.v1.where(
        utils.is_label_valid(labels), labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        utils.is_label_valid(labels), logits, tf.zeros_like(logits))

    epsilon_ = constant_op.constant(K.epsilon(), dtype=logits.dtype.base_dtype)
    output = clip_ops.clip_by_value(logits, epsilon_, 1. - epsilon_)

    bce = labels * math_ops.log(output + K.epsilon())
    return tf.compat.v1.losses.compute_weighted_loss(losses=-bce)
