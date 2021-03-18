import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from re_ranking.network import MultivariateRankingNetwork


class _LocalRanking(tf.keras.layers.Layer):
    """Local ranking function in DLCM.

    Call Arguments:

    inputs: List of the following tensors:
      * seq_output: GRU sequence output `Tensor` of shape `[batch_size, seq_len, dim]`.
      * final_state: GRU final state `Tensor` of shape `[batch_size, dim]`.
    """

    def __init__(self,
                 units,
                 **kwargs):
        super(_LocalRanking, self).__init__(**kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            seq_output_shape = tensor_shape.TensorShape(input_shape[0])
            final_state_shape = tensor_shape.TensorShape(input_shape[1])
        else:
            raise ValueError('The type of input shape argument is not supported, got: %s' % type(input_shape))

        last_dim = tensor_shape.dimension_value(seq_output_shape[-1])
        if last_dim != tensor_shape.dimension_value(final_state_shape[-1]):
            raise ValueError('The last dimension of the two inputs should be equal.')

        self.w = self.add_weight(
            'W',
            shape=[last_dim, self.units, last_dim],
            initializer=initializers.get('glorot_uniform'),
            regularizer=regularizers.get(None),
            constraint=constraints.get(None),
            dtype=self.dtype,
            trainable=True)
        self.b = self.add_weight(
            'b',
            shape=[self.units, last_dim],
            initializer=initializers.get('zeros'),
            regularizer=regularizers.get(None),
            constraint=constraints.get(None),
            dtype=self.dtype,
            trainable=True)
        self.v = self.add_weight(
            'V',
            shape=[self.units, 1],
            initializer=initializers.get('glorot_uniform'),
            regularizer=regularizers.get(None),
            constraint=constraints.get(None),
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            seq_output, final_state = inputs
        else:
            raise ValueError('Unexpected inputs to %s with length at %d' % (self.__class__, len(inputs)))

        # shape [B, seq_len, units, dim]
        outputs = tf.tensordot(seq_output, self.w, [[2], [0]])
        outputs = tf.add(outputs, self.b)
        outputs = activations.get('tanh')(outputs)
        # shape [B, seq_len, units]
        outputs = K.batch_dot(outputs, final_state)
        # shape [B, seq_len, 1]
        outputs = tf.tensordot(outputs, self.v, [[2], [0]])
        return outputs

    def get_config(self):
        config = super(_LocalRanking, self).get_config()
        config.update({
            'units': self.units,
        })
        return config


class DLCMRankingNetwork(MultivariateRankingNetwork):
    """Deep Listwise Context Model scoring"""

    def __init__(self,
                 context_feature_columns=None,
                 example_feature_columns=None,
                 sparse_features=None,
                 batch_norm_moment=0.999,
                 dropout=0.5,
                 name='dlcm_ranking_network',
                 **kwargs):
        if not example_feature_columns:
            raise ValueError('example_feature_columns must not be empty.')
        super(DLCMRankingNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            sparse_features=sparse_features,
            name=name,
            **kwargs)
        self._batch_norm_moment = batch_norm_moment
        self._dropout = dropout

        self.forward_gru = GRU(units=64,
                               return_sequences=True,
                               return_state=True)
        self.backward_gru = GRU(units=64,
                                return_sequences=True,
                                return_state=True,
                                go_backwards=True)
        self.score_layers = [BatchNormalization(momentum=self._batch_norm_moment),
                             Dense(1024, activation=tf.keras.activations.relu),
                             BatchNormalization(momentum=self._batch_norm_moment),
                             Dropout(self._dropout),

                             Dense(512, activation=tf.keras.activations.relu),
                             BatchNormalization(momentum=self._batch_norm_moment),
                             Dropout(self._dropout),

                             Dense(256, activation=tf.keras.activations.relu),
                             BatchNormalization(momentum=self._batch_norm_moment),
                             Dropout(self._dropout),

                             Dense(1, name='score')]

    def score(self,
              input_features=None,
              mask=None,
              training=None):
        """Scores context and examples to return a score per example."""

        # DLCM core model
        forward_seq_output, forward_final_state = self.forward_gru(input_features, mask=mask, training=training)
        backward_seq_output, backward_final_state = self.backward_gru(input_features, mask=mask, training=training)
        backward_seq_output = tf.reverse(backward_seq_output, axis=[1])
        seq_output = tf.concat([forward_seq_output, backward_seq_output], axis=-1)
        final_state = tf.concat([forward_final_state, backward_final_state], axis=-1)
        final_state = tf.expand_dims(final_state, axis=1)
        final_state = tf.tile(final_state, [1, tf.shape(seq_output)[1], 1])
        outputs = tf.concat([seq_output, final_state], axis=-1)
        for layer in self.score_layers:
            outputs = layer(outputs, training=training)
        return outputs

    def get_config(self):
        config = super(DLCMRankingNetwork, self).get_config()
        config.update({
            'dropout': self._dropout,
            'batch_norm_moment': self._batch_norm_moment
        })
        return config
