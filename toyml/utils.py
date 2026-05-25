from toyml.features import DenseFeature
from toyml.features import SequenceFeature
from toyml.features import SparseFeature
from tensorflow import keras as keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
import tensorflow as tf


def build_input_layer(feature):
    if isinstance(feature, SparseFeature):
        return Input(shape=(1,), name=feature.feature_name, dtype=feature.dtype)
    elif isinstance(feature, DenseFeature):
        return Input(shape=(1,), name=feature.feature_name, dtype=feature.dtype)
    elif isinstance(feature, SequenceFeature):
        return Input(shape=(feature.seq_len,),
                     name=feature.feature_name,
                     dtype=feature.element_sparse_feature.dtype)
    else:
        raise TypeError('Invalid feature column type {}'.format(feature))


def build_embedding_layer(input_dim, output_dim, name, mask_zero=False, regularizer=keras.regularizers.l2(1e-6)):
    return Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        embeddings_initializer=keras.initializers.GlorotUniform(),
        embeddings_regularizer=regularizer,
        mask_zero=mask_zero,
        name=name)


def expand_to_list_size(tensor, list_size):
    """Expand a [batch_size, ...] tensor to [batch_size, list_size, ...]."""
    return tf.tile(tf.expand_dims(tensor, axis=1), [1, list_size, 1])
