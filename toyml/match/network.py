import abc
from typing import Dict
from typing import Set

import six
import tensorflow as tf
from tensorflow_ranking.python import utils
from tensorflow_ranking.python.keras.network import RankingNetwork

from toyml.utils import build_embedding_layer
from toyml.features import SequenceFeature


class TwoTowerNetwork(RankingNetwork):
    """Base class for two tower network."""

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 context_feature_columns=None,
                 example_feature_columns=None,
                 sparse_features: Dict = None,
                 sequence_features: Dict = None,
                 name='two_tower_network',
                 **kwargs):
        super(TwoTowerNetwork, self).__init__(
            context_feature_columns=context_feature_columns,
            example_feature_columns=example_feature_columns,
            name=name,
            **kwargs)
        sparse_embed_layers = {}
        for name, feat in sparse_features.items():
            sparse_embed_layers[name] = build_embedding_layer(feat.vocab_size,
                                                              feat.embed_dim,
                                                              'embed_' + feat.feature_name)
        self._sparse_embed_layers = sparse_embed_layers
        self._sequence_features = sequence_features

    @abc.abstractmethod
    def score(self,
              context_inputs=None,
              example_inputs=None,
              mask=None,
              training=None):
        """Multivariate scoring of context and multi example to generate a list of score .

        Args:
          context_inputs: (dict) context inputs to 2D tensors of shape [batch_size, ...].
          example_inputs: (dict) example inputs names to 3D tensors of shape [batch_size, list_size, ...].
          mask: Mask is a tensor of shape [batch_size, list_size].
          training: (bool) whether in training or inference mode.

        Returns:
          (tf.Tensor) A score tensor of shape [batch_size, list_size].
        """
        raise NotImplementedError('Calling an abstract method, '
                                  'MultivariateRankingNetwork.score().')

    def compute_logits(self,
                       context_features=None,
                       example_features=None,
                       training=None,
                       mask=None):
        """Scores context and examples to return a score per example.

        Args:
          context_features: (dict) context feature names to 2D tensors of shape
            [batch_size, feature_dims].
          example_features: (dict) example feature names to 3D tensors of shape
            [batch_size, list_size, feature_dims].
          training: (bool) whether in train or inference mode.
          mask: (tf.Tensor) Mask is a tensor of shape [batch_size, list_size], which
            is True for a valid example and False for invalid one. If mask is None,
            all entries are valid.

        Returns:
          (tf.Tensor) A score tensor of shape [batch_size, list_size].

        Raises:
          ValueError: If `scorer` does not return a scalar output.

        """

        if not example_features:
            raise ValueError('Need a valid example feature.')

        tensor = next(six.itervalues(example_features))
        batch_size = tf.shape(tensor)[0]
        list_size = tf.shape(tensor)[1]
        if mask is None:
            mask = tf.ones(shape=[batch_size, list_size], dtype=tf.bool)
        nd_indices, nd_mask = utils.padded_nd_indices(is_valid=mask)

        # # Expand context features to be of [batch_size, list_size, ...].
        # batch_context_features = {}
        # for name, tensor in six.iteritems(context_features):
        #     x = tf.expand_dims(input=tensor, axis=1)
        #     x = tf.gather(x, tf.zeros([list_size], tf.int32), axis=1)
        #     batch_context_features[name] = utils.reshape_first_ndims(
        #         x, 2, [batch_size, list_size])
        batch_context_features = context_features

        batch_example_features = {}
        for name, tensor in six.iteritems(example_features):
            # Replace invalid example features with valid ones.
            padded_tensor = tf.gather_nd(tensor, nd_indices)
            batch_example_features[name] = utils.reshape_first_ndims(
                padded_tensor, 2, [batch_size, list_size])

        context_sparse_inputs, context_dense_inputs = [], []
        for name in batch_context_features:
            if name in self._sequence_features:
                feat = self._sequence_features[name]
                element_feat_name = feat.element_sparse_feature.feat_name
                if element_feat_name in self._sparse_embed_layers:
                    embed = self._sparse_embed_layers[element_feat_name](batch_context_features[name])
                    pooling = tf.keras.layers.GlobalAveragePooling1D()(embed)
                    context_sparse_inputs.append(pooling)
            elif name in self._sparse_embed_layers:
                context_sparse_inputs.append(self._sparse_embed_layers[name](batch_context_features[name]))
            else:
                context_dense_inputs.append(batch_context_features[name])
        context_sparse_inputs = [tf.squeeze(inpt, axis=2) for inpt in context_sparse_inputs]
        context_inputs = tf.concat(context_sparse_inputs + context_dense_inputs, axis=-1)

        example_sparse_inputs, example_dense_inputs = [], []
        for name in batch_example_features:
            if name in self._sparse_embed_layers:
                example_sparse_inputs.append(self._sparse_embed_layers[name](batch_example_features[name]))
            else:
                example_dense_inputs.append(batch_example_features[name])
        example_sparse_inputs = [tf.squeeze(inpt, axis=2) for inpt in example_sparse_inputs]
        example_inputs = tf.concat(example_sparse_inputs + example_dense_inputs, axis=-1)

        scores = self.score(context_inputs,
                            example_inputs,
                            nd_mask,
                            training=training)
        scores = tf.reshape(scores, shape=[batch_size, list_size, -1])

        # Apply nd_mask to zero out invalid entries.
        # Expand dimension and use broadcasting for filtering.
        expanded_nd_mask = tf.expand_dims(nd_mask, axis=2)
        scores = tf.where(expanded_nd_mask, scores, tf.zeros_like(scores))
        # Remove last dimension of shape = 1.
        try:
            logits = tf.squeeze(scores, axis=2)
        except:
            raise ValueError('Logits not of shape: [batch_size, list_size, 1]. '
                             'This could occur if the `scorer` does not return '
                             'a scalar output.')
        return logits
