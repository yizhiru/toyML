from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax

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
                 gamma=1.0,
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
        user_embed = context_inputs
        for layer in self._user_embed_layers:
            user_embed = layer(user_embed, training=training)
        item_embed = example_inputs
        for layer in self._item_embed_layers:
            item_embed = layer(item_embed, training=training)

        outputs = tf.einsum('bf,blf -> bl', user_embed, item_embed)
        scores = Softmax()(outputs)
        return scores

    def get_config(self):
        config = super(SBCNMNetwork, self).get_config()
        config.update({
            'hidden_layer_dims': self._hidden_layer_dims,
            'activation': self._activation
        })
        return config
