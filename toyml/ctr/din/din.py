from typing import List

from tensorflow import keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import PReLU

from toyml.features import DenseFeature
from toyml.features import SequenceFeature
from toyml.features import SparseFeature
from toyml.utils import build_embedding_layer
from toyml.utils import build_input_layer


def create_din_model(sparse_features: List[SparseFeature],
                     dense_features: List[DenseFeature],
                     candidate_features: List[SparseFeature],
                     hist_features: List[SequenceFeature],
                     dropout_rate=0.5):
    """
    Create deep interest network model

    :param sparse_features: 离散特征
    :param dense_features: 连续特征
    :param candidate_features: 候选item特征
    :param hist_features: 行为序列特征
    :param dropout_rate: dropout rate
    """

    """model inputs"""
    # sparse input layers
    sparse_input_list = [build_input_layer(feat) for feat in sparse_features]
    # candidate input layers
    candidate_input_list = [build_input_layer(feat) for feat in candidate_features]
    # dense input layers
    dense_input_list = [build_input_layer(feat) for feat in dense_features]
    # history input layer
    hist_input_list = [build_input_layer(feat) for feat in hist_features]

    input_list = sparse_input_list + candidate_input_list + dense_input_list + hist_input_list

    # sparse feature
    sparse_embed_list = [
        build_embedding_layer(feat.vocab_size, feat.embed_dim, 'embed_' + feat.feature_name)
        (sparse_input_list[i])
        for i, feat in enumerate(sparse_features)]
    # candidate item
    shared_embedding_layers = [
        build_embedding_layer(feat.vocab_size, feat.embed_dim, 'embed_' + feat.feature_name, feat.mask_zero)
        for feat in candidate_features]
    candidate_feat_num = len(candidate_features)
    candidate_embed_list = []
    for j in range(candidate_feat_num):
        layer = shared_embedding_layers[j](candidate_input_list[j])
        candidate_embed_list.append(layer)
    candidate_embed = Concatenate(name='candidate_concat')(candidate_embed_list)

    # history attention
    act_num = len(hist_input_list) // candidate_feat_num
    hist_attention_list = []
    for i in range(act_num):
        per_act_list = []
        for j in range(candidate_feat_num):
            layer = shared_embedding_layers[j](hist_input_list[i * candidate_feat_num + j])
            per_act_list.append(layer)
        per_act_embed = Concatenate(name=f'act{i}_concat')(per_act_list)
        # history attention
        per_act_attention = Attention(name=f'act{i}_attention')([candidate_embed, per_act_embed])
        hist_attention_list.append(per_act_attention)

    # concat features
    sparse_inputs = Flatten(name='sparse_inputs')(Concatenate(name='sparse_concat')(sparse_embed_list + [candidate_embed]))
    dense_inputs = Concatenate(name='dense_inputs')(dense_input_list)
    if len(hist_attention_list) == 1:
        hist_layer = hist_attention_list[0]
    else:
        hist_layer = Concatenate(name='attention_concat')(hist_attention_list)
    hist_inputs = Flatten(name='hist_inputs')(hist_layer)
    concat_inputs = Concatenate(name='concat_inputs')([sparse_inputs, dense_inputs, hist_inputs])
    dense_layer1 = Dense(200, name='dense1')(concat_inputs)
    prelu_layer1 = PReLU(name='prelu1')(dense_layer1)
    dropout_layer1 = Dropout(rate=dropout_rate, name='dropout1')(prelu_layer1)
    dense_layer2 = Dense(80, name='dense2')(dropout_layer1)
    prelu_layer2 = PReLU(name='prelu2')(dense_layer2)
    dropout_layer2 = Dropout(rate=dropout_rate, name='dropout2')(prelu_layer2)
    output = Dense(1, activation=K.sigmoid, name='dense3')(dropout_layer2)

    model = keras.Model(inputs=input_list, outputs=output)
    return model
