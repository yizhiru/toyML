from typing import Dict

import six
import tensorflow as tf
from tensorflow_ranking.python import utils
from tensorflow_ranking.python.keras.network import RankingNetwork

from utils import build_embedding_layer


class DNNRankingNetwork(RankingNetwork):
    """"Deep Neural Network (DNN) scoring."""

    def __init__(self,
                 context_feature_columns=None,
                 example_feature_columns=None,
                 sparse_features: Dict = None,
                 hidden_layer_dims=None,
                 activation=None,
                 use_batch_norm=True,
                 batch_norm_moment=0.999,
                 dropout=0.5,
                 name='dnn_ranking_network',
                 **kwargs):
        if not example_feature_columns or not hidden_layer_dims:
            raise ValueError('example_feature_columns or hidden_layer_dims must not be empty.')
        super(DNNRankingNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            name=name,
            **kwargs)
        self._hidden_layer_dims = [int(d) for d in hidden_layer_dims]
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._batch_norm_moment = batch_norm_moment
        self._dropout = dropout

        sparse_embed_layers = {}
        for name, feat in sparse_features.items():
            sparse_embed_layers[name] = build_embedding_layer(feat.vocab_size,
                                                              feat.embed_dim,
                                                              'embed_' + feat.feature_name)
        self._sparse_embed_layers = sparse_embed_layers
        layers = []
        if self._use_batch_norm:
            layers.append(
                tf.keras.layers.BatchNormalization(momentum=self._batch_norm_moment))
        for _, layer_width in enumerate(self._hidden_layer_dims):
            layers.append(tf.keras.layers.Dense(units=layer_width))
            if self._use_batch_norm:
                layers.append(
                    tf.keras.layers.BatchNormalization(
                        momentum=self._batch_norm_moment))
            layers.append(tf.keras.layers.Activation(activation=self._activation))
            layers.append(tf.keras.layers.Dropout(rate=self._dropout))
        self._scoring_layers = layers
        self._output_score_layer = tf.keras.layers.Dense(units=1)

    def compute_logits(self,
                       context_features=None,
                       example_features=None,
                       training=True,
                       mask=None):
        tensor = next(six.itervalues(example_features))
        batch_size = tf.shape(tensor)[0]
        list_size = tf.shape(tensor)[1]
        if mask is None:
            mask = tf.ones(shape=[batch_size, list_size], dtype=tf.bool)
        nd_indices, nd_mask = utils.padded_nd_indices(is_valid=mask)

        # Expand query features to be of [batch_size, list_size, ...].
        large_batch_context_features = {}
        for name, tensor in six.iteritems(context_features):
            x = tf.expand_dims(input=tensor, axis=1)
            x = tf.gather(x, tf.zeros([list_size], tf.int32), axis=1)
            large_batch_context_features[name] = utils.reshape_first_ndims(
                x, 2, [batch_size * list_size])

        large_batch_example_features = {}
        for name, tensor in six.iteritems(example_features):
            # Replace invalid example features with valid ones.
            padded_tensor = tf.gather_nd(tensor, nd_indices)
            large_batch_example_features[name] = utils.reshape_first_ndims(
                padded_tensor, 2, [batch_size * list_size])

        # Get scores for large batch.
        sparse_input, dense_input = [], []
        for name in large_batch_context_features:
            if name in self._sparse_embed_layers:
                sparse_input.append(self._sparse_embed_layers[name](large_batch_context_features[name]))
            else:
                dense_input.append(context_features[name])
        for name in large_batch_example_features:
            if name in self._sparse_embed_layers:
                sparse_input.append(self._sparse_embed_layers[name](large_batch_example_features[name]))
            else:
                dense_input.append(large_batch_example_features[name])
        sparse_input = [tf.keras.layers.Flatten()(inpt) for inpt in sparse_input]

        inputs = tf.concat(sparse_input + dense_input, 1)
        outputs = inputs
        for layer in self._scoring_layers:
            outputs = layer(outputs, training=training)

        scores = self._output_score_layer(outputs, training=training)
        logits = tf.reshape(
            scores, shape=[batch_size, list_size])

        # Apply nd_mask to zero out invalid entries.
        logits = tf.where(nd_mask, logits, tf.zeros_like(logits))
        return logits

    def get_config(self):
        config = super(DNNRankingNetwork, self).get_config()
        config.update({
            'hidden_layer_dims': self._hidden_layer_dims,
            'activation': self._activation,
            'use_batch_norm': self._use_batch_norm,
            'batch_norm_moment': self._batch_norm_moment,
            'dropout': self._dropout,
        })
        return config
