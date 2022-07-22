import os
from typing import List

import tensorflow as tf
import tensorflow_ranking as tfr
from absl import flags
from tensorflow import feature_column as fc
from tensorflow import keras
from tensorflow_ranking.python.keras.metrics import MeanAveragePrecisionMetric
from tensorflow_ranking.python.keras.metrics import NDCGMetric
from tensorflow_ranking.python.keras.metrics import PrecisionMetric

from toyml.features import DenseFeature
from toyml.features import SequenceFeature
from toyml.features import SparseFeature
from toyml.match import DSSMNetwork
from toyml.match import dssm_loss

flags.DEFINE_string("train_path", 'train.tfrecord', "Input file path used for training.")
flags.DEFINE_string("eval_path", 'eval.tfrecord', "Input file path used for eval.")
flags.DEFINE_string("model_dir", 'dnn_model', "Output directory for models.")

flags.DEFINE_integer("batch_size", 1000, "The batch size for train.")
flags.DEFINE_integer("epochs", 10, "Number of epochs for train.")

flags.DEFINE_float("learning_rate", 0.05, "Learning rate for optimizer.")
flags.DEFINE_integer("list_size", 15, "List size used for training. Use None for dynamic list size.")

flags.DEFINE_string('device_map', '2', 'CUDA visible devices.')

FLAGS = flags.FLAGS

_LABEL = 'label'
_SIZE = "example_list_size"

_EMBEDDING_DIM = 12


def _create_feature_columns():
    def _sparse_fc(sparse_feature: SparseFeature):
        return fc.numeric_column(key=sparse_feature.feature_name, dtype=tf.int64)

    def _dense_fc(dense_feature: DenseFeature):
        return fc.numeric_column(key=dense_feature.feature_name)

    def _sequence_fc(sequence_feature: SequenceFeature):
        return fc.numeric_column(key=sequence_feature.feature_name,
                                 shape=(sequence_feature.seq_len,),
                                 dtype=tf.int64)

    def _all_fc(sparse_features: List[SparseFeature],
                dense_features: List[DenseFeature],
                sequence_features: List[SequenceFeature] = None):
        feature_columns = {}
        for feat in sparse_features:
            feature_columns[feat.feature_name] = _sparse_fc(feat)
        for feat in dense_features:
            feature_columns[feat.feature_name] = _dense_fc(feat)
        if sequence_features is not None:
            for feat in sequence_features:
                feature_columns[feat.feature_name] = _sequence_fc(feat)
        return feature_columns

    context_sparse_features, context_dense_features, context_sequence_features, \
    example_sparse_features, example_dense_features = _create_match_features()

    context_feature_columns = _all_fc(context_sparse_features, context_dense_features, context_sequence_features)
    example_feature_columns = _all_fc(example_sparse_features, example_dense_features)
    return context_feature_columns, example_feature_columns


def _create_match_features():
    # context features
    context_sparse_features = [
        SparseFeature('u_age', 8, _EMBEDDING_DIM),
        SparseFeature('u_gender', 3, _EMBEDDING_DIM),
    ]
    context_dense_features = [
        DenseFeature('u_ctr_30d'),
    ]
    context_sequence_features = [
        SequenceFeature('click_item_hist', 50, SparseFeature('item_id', 100, _EMBEDDING_DIM)),
    ]

    # example features
    example_sparse_features = [
        SparseFeature('i_hash_id', 100, _EMBEDDING_DIM),
    ]
    example_dense_features = [
        DenseFeature('i_ctr_7d'),
    ]

    return context_sparse_features, context_dense_features, context_sequence_features, \
           example_sparse_features, example_dense_features


def make_dataset(file_pattern,
                 batch_size,
                 randomize_input=True,
                 num_epochs=1):
    context_feature_columns, example_feature_columns = _create_feature_columns()
    context_feature_spec = fc.make_parse_example_spec(
        context_feature_columns.values())
    label_column = tf.feature_column.numeric_column(_LABEL, dtype=tf.int64, default_value=-1)
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns.values()) + [label_column])
    dataset = tfr.data.build_ranking_dataset(
        file_pattern=file_pattern,
        data_format=tfr.data.SEQ,
        batch_size=batch_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        list_size=FLAGS.list_size,
        reader=tf.data.TFRecordDataset,
        reader_args=['GZIP', 32],
        shuffle=randomize_input,
        num_epochs=num_epochs,
        size_feature_name=_SIZE)

    def _separate_features_and_label(features):
        label = tf.squeeze(features.pop(_LABEL), axis=2)
        label = tf.cast(label, tf.float32)
        return features, label

    dataset = dataset.map(_separate_features_and_label)
    return dataset


def train_and_eval():
    context_feature_columns, example_feature_columns = _create_feature_columns()
    train_dataset = make_dataset(FLAGS.train_path, FLAGS.batch_size)
    eval_dataset = make_dataset(FLAGS.eval_path, FLAGS.batch_size)

    context_sparse_features, _, context_sequence_features, example_sparse_features, _ = _create_match_features()
    # parse sparse features
    sparse_features = {}
    for feat in context_sparse_features + example_sparse_features:
        sparse_features[feat.feature_name] = feat
    for feat in context_sequence_features:
        element_feat = feat.element_sparse_feature
        sparse_features[element_feat.feature_name] = element_feat
    # parse sequence features
    sequence_features = {feat.feature_name: feat for feat in context_sequence_features}

    network = DSSMNetwork(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        sparse_features=sparse_features,
        sequence_features=sequence_features,
        hidden_layer_dims=[1024, 512, 256],
        activation=tf.nn.relu)

    metrics = [PrecisionMetric(name='precision@5', topn=5),
               MeanAveragePrecisionMetric(name='map@5', topn=5),
               NDCGMetric(name='ndcg@5', topn=5)]
    ranker = tfr.keras.model.create_keras_model(
        network=network,
        loss=dssm_loss,
        metrics=metrics,
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        size_feature_name=_SIZE)
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_dir,
                                                 monitor='val_precision@5',
                                                 mode='max',
                                                 save_best_only=True)
                 ]
    ranker.fit(train_dataset,
               validation_data=eval_dataset,
               epochs=FLAGS.epochs,
               callbacks=callbacks)


def main(_):
    tf.compat.v1.set_random_seed(1234)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

    train_and_eval()


if __name__ == "__main__":
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("eval_path")

    tf.compat.v1.app.run()
