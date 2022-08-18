from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow_ranking.python import utils

from toyml.match import cosine_similarity
from toyml.match.network import TwoTowerNetwork


class SBCNMNetwork(TwoTowerNetwork):
    """SBCNM network for match"""

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
        super(SBCNMNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            sparse_features=sparse_features,
            sequence_features=sequence_features,
            name=name,
            **kwargs)
        self._hidden_layer_dims = [int(d) for d in hidden_layer_dims]
        self._activation = activation

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

        outputs = cosine_similarity(user_embed, item_embed)
        scores = Softmax()(outputs)
        return scores

    def get_config(self):
        config = super(SBCNMNetwork, self).get_config()
        config.update({
            'hidden_layer_dims': self._hidden_layer_dims,
            'activation': self._activation
        })
        return config
