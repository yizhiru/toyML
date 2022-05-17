import os
import shutil

import tensorflow as tf
import tensorflow_ranking as tfr
from absl import flags
from tensorflow.compat.v1 import feature_column as fc

flags.DEFINE_enum(
    "data_format", "example_list_with_context",
    ["example_list_with_context", "example_in_example", "sequence_example"],
    "Data format defined in data.py.")
flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for eval.")
flags.DEFINE_string("test_path", None, "Input file path used for test.")

flags.DEFINE_string("model_dir", None, "Output directory for models.")
flags.DEFINE_string('pb_dir', None, 'Output directory for pb models.')

flags.DEFINE_integer("batch_size", 32, "The batch size for train.")
flags.DEFINE_integer("num_train_steps", 15000, "Number of steps for train.")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.8, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["64", "32", "16"],
                  "Sizes for hidden layers.")
flags.DEFINE_integer(
    "list_size", None,
    "List size used for training. Use None for dynamic list size.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")
flags.DEFINE_string("loss", "approx_ndcg_loss",
                    "The RankingLossKey for the loss function.")
flags.DEFINE_string("weights_feature_name", "",
                    "The name of the feature where unbiased learning-to-rank "
                    "weights are stored.")
flags.DEFINE_bool("listwise_inference", False,
                  "If true, exports accept `data_format` while serving.")

flags.DEFINE_string('device_map', '-1', 'CUDA visible devices.')

FLAGS = flags.FLAGS

_LABEL_FEATURE = "label"
_PADDING_LABEL = -1
_EMBEDDING_DIMENSION = 20

col_names = ['u_device_os', 'position']
feature_names = ['u_device_os', 'position']


def example_feature_columns():
    """Returns the example feature columns."""
    feature_dict = {}
    # user features
    feature_dict['u_device_os'] = fc.embedding_column(
        fc.categorical_column_with_vocabulary_list("u_device_os", ['iphone', 'android', '-1']), 5)

    # shop features
    # item features

    feature_dict['position'] = fc.embedding_column(fc.categorical_column_with_hash_bucket('position', 500), 5)
    return feature_dict


def make_input_fn(file_pattern,
                  batch_size,
                  randomize_input=True,
                  num_epochs=None):
    """Returns `Estimator` `input_fn` for TRAIN and EVAL.
    Args:
      file_pattern: (string) file pattern for the TFRecord input data.
      batch_size: (int) number of input examples to process per batch.
      randomize_input: (bool) if true, randomize input example order. It should
        almost always be true except for unittest/debug purposes.
      num_epochs: (int) Number of times the input dataset must be repeated. None
        to repeat the data indefinitely.
    Returns:
      An `input_fn` for `Estimator`.
    """
    tf.compat.v1.logging.info("FLAGS.data_format={}".format(FLAGS.data_format))

    def _input_fn():
        """Defines the input_fn."""
        label_column = tf.feature_column.numeric_column(
            _LABEL_FEATURE, dtype=tf.float32, default_value=_PADDING_LABEL)
        example_feature_spec = tf.feature_column.make_parse_example_spec(
            list(example_feature_columns().values()) + [label_column])
        dataset = tfr.data.build_ranking_dataset(
            file_pattern=file_pattern,
            data_format=FLAGS.data_format,
            batch_size=batch_size,
            list_size=FLAGS.list_size,
            context_feature_spec=None,
            example_feature_spec=example_feature_spec,
            reader=tf.data.TFRecordDataset,
            shuffle=randomize_input,
            num_epochs=num_epochs)
        features = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
        label = tf.cast(label, tf.float32)

        return features, label

    return _input_fn


def make_serving_input_fn():
    """Returns serving input fn."""
    # context_feature_spec = tf.feature_column.make_parse_example_spec(
    #     context_feature_columns().values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        example_feature_columns().values())
    if FLAGS.listwise_inference:
        # Exports accept the specificed FLAGS.data_format during serving.
        return tfr.data.build_ranking_serving_input_receiver_fn(
            data_format=FLAGS.data_format,
            context_feature_spec=None,
            example_feature_spec=example_feature_spec)
    elif FLAGS.group_size == 1:
        # Exports accept tf.Example when group_size = 1.
        feature_spec = {}
        feature_spec.update(example_feature_spec)
        # feature_spec.update(context_feature_spec)
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec)
    else:
        raise ValueError("FLAGS.group_size should be 1, but is {} when "
                         "FLAGS.export_listwise_inference is False".format(
            FLAGS.group_size))


def input_recv_fn():
    input_dict = {
        name: tf.compat.v1.placeholder(tf.string, shape=(None, 1, 1), name=name) for name in feature_names}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(input_dict)()


def make_transform_fn():
    """Returns a transform_fn that converts features to dense Tensors."""

    def _transform_fn(features, mode):
        """Defines transform_fn."""
        if mode == tf.estimator.ModeKeys.PREDICT and not FLAGS.listwise_inference:
            # We expect tf.Example as input during serving. In this case, group_size
            # must be set to 1.
            if FLAGS.group_size != 1:
                raise ValueError(
                    "group_size should be 1 to be able to export model, but get %s" %
                    FLAGS.group_size)
            context_features, example_features = (
                tfr.feature.encode_pointwise_features(
                    features=features,
                    context_feature_columns=None,
                    example_feature_columns=example_feature_columns(),
                    mode=mode,
                    scope="transform_layer"))
        else:
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=None,
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer")

        return context_features, example_features

    return _transform_fn


def make_score_fn():
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a group of documents."""
        del [params, config]
        with tf.compat.v1.name_scope("input_layer"):
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(example_feature_columns())
            ]
            input_layer = tf.concat(group_input, 1)
            tf.compat.v1.summary.scalar("input_sparsity",
                                        tf.nn.zero_fraction(input_layer))
            tf.compat.v1.summary.scalar("input_max",
                                        tf.reduce_max(input_tensor=input_layer))
            tf.compat.v1.summary.scalar("input_min",
                                        tf.reduce_min(input_tensor=input_layer))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = input_layer
        cur_layer = tf.compat.v1.layers.batch_normalization(
            cur_layer, training=is_training, momentum=0.99)

        for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training, momentum=0.99)
            cur_layer = tf.nn.relu(cur_layer)
            tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                        tf.nn.zero_fraction(cur_layer))
            cur_layer = tf.compat.v1.layers.dropout(
                inputs=cur_layer, rate=FLAGS.dropout_rate, training=is_training)
        logits = tf.compat.v1.layers.dense(cur_layer, units=FLAGS.group_size)
        return logits

    return _score_fn


def eval_metric_fns():
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
            tfr.metrics.RankingMetricKey.ARP,
            tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
        ]
    })
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [5, 10, 20, 50]
    })
    # for topn in [5, 10, 20, 50]:
    #     metric_fns["metric/weighted_ndcg@%d" % topn] = (
    #         tfr.metrics.make_ranking_metric_fn(
    #             tfr.metrics.RankingMetricKey.NDCG,
    #             weights_feature_name=FLAGS.weights_feature_name, topn=topn))
    return metric_fns


def train_and_eval():
    """Train and Evaluate."""
    train_input_fn = make_input_fn(FLAGS.train_path, FLAGS.batch_size)
    eval_input_fn = make_input_fn(
        FLAGS.vali_path, FLAGS.batch_size, randomize_input=False, num_epochs=1)
    test_input_fn = make_input_fn(
        FLAGS.test_path, FLAGS.batch_size, randomize_input=False, num_epochs=1)

    optimizer = tf.compat.v1.train.AdagradOptimizer(
        learning_rate=FLAGS.learning_rate)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([minimize_op, update_ops])
        return train_op

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(
            FLAGS.loss,
            weights_feature_name=FLAGS.weights_feature_name),
        eval_metric_fns=eval_metric_fns(),
        train_op_fn=_train_op_fn)

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=FLAGS.group_size,
            transform_fn=make_transform_fn(),
            ranking_head=ranking_head),
        model_dir=FLAGS.model_dir,
        config=tf.estimator.RunConfig(save_checkpoints_steps=1000))

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    exporters = tf.estimator.LatestExporter(
        "saved_model_exporter", serving_input_receiver_fn=make_serving_input_fn())

    eval_spec = tf.estimator.EvalSpec(
        name="eval",
        input_fn=eval_input_fn,
        steps=1,
        exporters=exporters,
        start_delay_secs=0,
        throttle_secs=15)

    # Train and validate.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Test
    estimator.evaluate(input_fn=test_input_fn)

    # export pb model
    estimator.export_saved_model(FLAGS.pb_dir, input_recv_fn)


def main(_):
    tf.compat.v1.set_random_seed(1234)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_map

    # rm exists ckpt model
    if os.path.exists(FLAGS.model_dir):
        shutil.rmtree(FLAGS.model_dir)
    os.makedirs(FLAGS.model_dir)

    train_and_eval()


if __name__ == "__main__":
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("vali_path")
    flags.mark_flag_as_required("test_path")
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("pb_dir")

    tf.compat.v1.app.run()
