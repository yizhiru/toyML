from typing import List

import tensorflow as tf
import tensorflow_ranking as tfr
from absl import flags
from tensorflow import feature_column as fc
from tensorflow import keras

from tensorflow_ranking.python.keras.metrics import NDCGMetric
from tensorflow_ranking.python.keras.metrics import PrecisionMetric
from tensorflow_ranking.python.keras.metrics import MeanAveragePrecisionMetric

from features import DenseFeature
from features import SparseFeature
from re_ranking import prm
import os

flags.DEFINE_string("train_path", 'train.tfrecord', "Input file path used for training.")
flags.DEFINE_string("eval_path", 'eval.tfrecord', "Input file path used for eval.")
flags.DEFINE_string("model_dir", 'prm_model', "Output directory for models.")

flags.DEFINE_integer("batch_size", 1000, "The batch size for train.")
flags.DEFINE_integer("epochs", 6, "Number of epochs for train.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_integer("list_size", 15, "List size used for training. Use None for dynamic list size.")

flags.DEFINE_string('device_map', '2', 'CUDA visible devices.')

FLAGS = flags.FLAGS

_LABEL = 'is_click'
_SIZE = "example_list_size"

_EMBEDDING_DIM = 12


def _create_feature_columns():
    def _sparse_fc(sparse_feature: SparseFeature):
        return fc.numeric_column(key=sparse_feature.feature_name, dtype=tf.int64)

    def _dense_fc(dense_feature: DenseFeature):
        return fc.numeric_column(key=dense_feature.feature_name)

    def _all_fc(sparse_features: List[SparseFeature],
                dense_features: List[DenseFeature]):
        feature_columns = {}
        for feat in sparse_features:
            feature_columns[feat.feature_name] = _sparse_fc(feat)
        for feat in dense_features:
            feature_columns[feat.feature_name] = _dense_fc(feat)
        return feature_columns

    context_sparse_features, context_dense_features, \
    example_sparse_features, example_dense_features = _create_ranking_features()

    context_feature_columns = _all_fc(context_sparse_features, context_dense_features)
    example_feature_columns = _all_fc(example_sparse_features, example_dense_features)
    return context_feature_columns, example_feature_columns


def _create_ranking_features():
    # context features
    context_sparse_features = [
        SparseFeature('u_age', 8, _EMBEDDING_DIM),
        SparseFeature('u_gender', 3, _EMBEDDING_DIM),
    ]
    context_dense_features = []

    # example features
    example_sparse_features = [
        SparseFeature('i_hash_cid1', 100, _EMBEDDING_DIM),
        SparseFeature('i_hash_cid4', 5000, _EMBEDDING_DIM),
        SparseFeature('i_price_grade', 11, _EMBEDDING_DIM),
    ]
    example_dense_features = [
        DenseFeature('i_ctr_7d'),
        DenseFeature('i_ctr_30d'),
    ]
    return context_sparse_features, context_dense_features, example_sparse_features, example_dense_features


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

    context_sparse_features, context_dense_features, \
    example_sparse_features, example_dense_features = _create_ranking_features()
    sparse_features = {}
    for feat in context_sparse_features + example_sparse_features:
        sparse_features[feat.feature_name] = feat
    network = prm.PRMRankingNetwork(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        sparse_features=sparse_features,
        dropout=0.5,
        encoder_blocks=4)

    metrics = [PrecisionMetric(name='precision@2', topn=2),
               PrecisionMetric(name='precision@5', topn=5),
               MeanAveragePrecisionMetric(name='map@2', topn=2),
               MeanAveragePrecisionMetric(name='map@5', topn=5),
               NDCGMetric(name='ndcg@2', topn=2)]
    ranker = tfr.keras.model.create_keras_model(
        network=network,
        loss=prm.negative_log_likelihood_loss,
        metrics=metrics,
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        size_feature_name=_SIZE)
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_dir,
                                                 monitor='val_precision@2',
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
    tf.compat.v1.app.run()
