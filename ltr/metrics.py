import numpy as np
import tensorflow as tf
from tensorflow_ranking.python import metrics


def tf_ndcg(y_true, y_pred, y_group, top_n=5):
    def to_tensor(y):
        list_size = max(y_group)
        begin_idx = 0
        grouped_y = []
        for g in y_group:
            one = np.pad(y[begin_idx: begin_idx+g], (0, max(0, list_size - g)), 'constant')
            grouped_y.append(one)
            begin_idx += g
        return tf.convert_to_tensor(np.array(grouped_y), dtype=float)

    y_true = to_tensor(y_true)
    y_pred = to_tensor(y_pred)
    n = metrics.normalized_discounted_cumulative_gain(y_true, y_pred, topn=top_n)
    with tf.Session() as sess:
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="normalized_discounted_cumulative_gain")
        running_initializer = tf.variables_initializer(var_list=running_vars)

        sess.run(tf.global_variables_initializer())
        sess.run(running_initializer)
        n = sess.run(n)
        return n[1]