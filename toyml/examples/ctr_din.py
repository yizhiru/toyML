import os

import tensorflow as tf
from absl import flags
from tensorflow import keras as keras

from toyml.ctr import create_din_model
from toyml.features import SparseFeature, DenseFeature, SequenceFeature

flags.DEFINE_string("train_path", 'train.tfrecord', "Input file path used for training.")
flags.DEFINE_string("eval_path", 'eval.tfrecord', "Input file path used for eval.")
flags.DEFINE_string('pb_dir', 'model', 'Output directory for pb models.')

flags.DEFINE_integer("batch_size", 5000, "The batch size for train.")
flags.DEFINE_integer("epochs", 10, "Number of epochs for train.")

flags.DEFINE_string('device_map', '0,1,2,3', 'CUDA visible devices.')

FLAGS = flags.FLAGS

_LABEL = "is_click"

# embedding dimension
_EMBEDDING_DIM = 12


def _create_example_features():
    click_len, collect_len, cart_len, order_len = 60, 20, 30, 40
    sparse_features = [
        SparseFeature('u_age', 8, _EMBEDDING_DIM),
        SparseFeature('u_gender', 3, _EMBEDDING_DIM),
        SparseFeature('hash_i_cid1', 100, _EMBEDDING_DIM),
        SparseFeature('i_price_grade', 100, _EMBEDDING_DIM),
        SparseFeature('s_province', 35, _EMBEDDING_DIM),
        SparseFeature('s_city', 340, _EMBEDDING_DIM),
        SparseFeature('u_cid1_price_prefer', 11, _EMBEDDING_DIM),
        SparseFeature('u_cid1_expose_cnt', 10, _EMBEDDING_DIM),
    ]

    candidate_features = [
        SparseFeature('hash_item_id', 1000000, _EMBEDDING_DIM, mask_zero=True),
        SparseFeature('hash_seller_id', 800000, _EMBEDDING_DIM, mask_zero=True),
    ]

    dense_features = [
        DenseFeature('u_ctr_7d'),
        DenseFeature('u_ctr_30d'),
        DenseFeature('i_ctr_7d'),
        DenseFeature('i_ctr_30d'),
        DenseFeature('s_ctr_7d'),
        DenseFeature('s_ctr_30d'),
    ]

    # 行为序列特征，长度为attention feature 整数倍
    hist_features = [
        SequenceFeature('click_item_hist', click_len, candidate_features[0]),
        SequenceFeature('click_seller_hist', click_len, candidate_features[1]),
    ]
    return sparse_features, dense_features, candidate_features, hist_features


def make_input_dataset(tfrecord_path: str, batch_size=1024, is_train=False):
    sparse_features, dense_features, candidate_features, hist_features = _create_example_features()
    sparse_desc_dict = {
        feat.feature_name: tf.io.FixedLenFeature([1], tf.int64)
        for feat in sparse_features}
    candidate_desc_dict = {
        feat.feature_name: tf.io.FixedLenFeature([1], tf.int64)
        for feat in candidate_features}
    dense_desc_dict = {
        feat.feature_name: tf.io.FixedLenFeature([1], tf.float32)
        for feat in dense_features}
    hist_desc_dict = {
        feat.feature_name: tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0)
        for feat in hist_features}
    label_desc = {_LABEL: tf.io.FixedLenFeature([1], tf.int64)}

    def _parse_example_fn(example_proto):
        example_dict = tf.io.parse_single_example(example_proto,
                                                  {**sparse_desc_dict, **candidate_desc_dict, **dense_desc_dict,
                                                   **hist_desc_dict, **label_desc})
        label = example_dict.pop(_LABEL)
        return example_dict, label

    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=32)
    if is_train:
        dataset = dataset.shuffle(buffer_size=20000)
    return dataset.map(_parse_example_fn, num_parallel_calls=32).batch(batch_size).prefetch(2)


def main(_):
    tf.compat.v1.set_random_seed(1234)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

    train_dataset = make_input_dataset(FLAGS.train_path, FLAGS.batch_size, is_train=True)
    eval_dataset = make_input_dataset(FLAGS.eval_path, 20000)

    sparse_features, dense_features, candidate_features, hist_features = _create_example_features()
    model = create_din_model(sparse_features,
                             dense_features,
                             candidate_features,
                             hist_features,
                             0.5)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.AUC()])
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=FLAGS.pb_dir,
                                                 monitor='val_auc',
                                                 mode='max',
                                                 save_best_only=True)
                 ]
    model.fit(train_dataset,
              validation_data=eval_dataset,
              epochs=FLAGS.epochs,
              callbacks=callbacks)


if __name__ == "__main__":
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("eval_path")
    flags.mark_flag_as_required("pb_dir")

    tf.compat.v1.app.run()
