from features import SparseFeature
from features import DenseFeature
from features import SequenceFeature
from tensorflow import keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding


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
        embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4),
        embeddings_regularizer=regularizer,
        mask_zero=mask_zero,
        name=name)
