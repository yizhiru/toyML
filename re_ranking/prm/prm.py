import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow_ranking.python import utils

from re_ranking.network import MultivariateRankingNetwork
from re_ranking.prm import TransformerEncoder


class PRMRankingNetwork(MultivariateRankingNetwork):
    """Personalized Re-ranking Model scoring"""

    def __init__(self,
                 context_feature_columns=None,
                 example_feature_columns=None,
                 sparse_features=None,
                 batch_norm_moment=0.999,
                 dropout=0.5,
                 encoder_blocks=4,
                 name='prm_ranking_network',
                 **kwargs):
        if not example_feature_columns:
            raise ValueError('example_feature_columns must not be empty.')
        super(PRMRankingNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            sparse_features=sparse_features,
            name=name,
            **kwargs)
        self._batch_norm_moment = batch_norm_moment
        self._dropout = dropout
        self._encoder_blocks = encoder_blocks

        self.encoder = TransformerEncoder(num_blocks=self._encoder_blocks,
                                          num_heads=2,
                                          ffn_dim=1024)
        self.score_layers = [
            Dense(1, activation=None),
        ]

    def score(self,
              input_features=None,
              mask=None,
              training=None):

        # transformer encoder
        outputs = self.encoder(input_features, mask=mask, training=training)
        expanded_nd_mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, tf.shape(outputs)[-1]])
        outputs = tf.where(expanded_nd_mask, outputs, tf.zeros_like(outputs))

        for layer in self.score_layers:
            outputs = layer(outputs, training=training)
        outputs = tf.squeeze(outputs, axis=-1)
        outputs = tf.keras.layers.Softmax(axis=-1)(outputs, mask=mask)
        return outputs

    def get_config(self):
        config = super(PRMRankingNetwork, self).get_config()
        config.update({
            'batch_norm_moment': self._batch_norm_moment,
            'dropout': self._dropout,
            'encoder_blocks': self._encoder_blocks
        })
        return config


def negative_log_likelihood_loss(labels, logits):
    """Computes the negative log likehood loss ."""
    labels = tf.compat.v1.where(
        utils.is_label_valid(labels), labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        utils.is_label_valid(labels), logits, tf.zeros_like(logits))

    epsilon_ = constant_op.constant(K.epsilon(), dtype=logits.dtype.base_dtype)
    output = clip_ops.clip_by_value(logits, epsilon_, 1. - epsilon_)

    bce = labels * math_ops.log(output + K.epsilon())
    # bce += (1 - labels) * math_ops.log(1 - output + K.epsilon())
    return tf.compat.v1.losses.compute_weighted_loss(losses=-bce)
