import tensorflow as tf
from tensorflow_serving.apis import input_pb2


def group_count(qid_list):
    qid = qid_list[0]
    grp_cnt = [0]
    for idx in qid_list:
        if idx == qid:
            grp_cnt[-1] = grp_cnt[-1] + 1
        else:
            qid = idx
            grp_cnt.append(1)
    return grp_cnt


def generate_tf_record(input_path, output_path, feature_names):
    """generate tfrecord"""

    def _parse_line(line):
        """Parses a single line in LibSVM format."""
        tokens = line.split()
        assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
        label = float(tokens[0])
        qid = tokens[1]
        features = {k: v for k, v in zip(feature_names, tokens[2:])}
        return qid, features, label

    def _generate_per_example(features, label):
        example_feature_dict = {k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode()])) for k, v in
                                features.items()}
        example_feature_dict['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
        return tf.train.Example(features=tf.train.Features(feature=example_feature_dict))

    tf.compat.v1.logging.info("Start to convert {} to {}".format(input_path, output_path))
    writer = tf.io.TFRecordWriter(output_path)
    with open(input_path, "rt") as f:
        qid_mark = ''
        elwc = input_pb2.ExampleListWithContext()
        for line in f:
            qid, features, label = _parse_line(line)
            if qid_mark == '':
                elwc.examples.add().CopyFrom(_generate_per_example(features, label))
                qid_mark = qid
            elif qid == qid_mark:
                elwc.examples.add().CopyFrom(_generate_per_example(features, label))
            else:
                writer.write(elwc.SerializeToString())
                elwc = input_pb2.ExampleListWithContext()
                elwc.examples.add().CopyFrom(_generate_per_example(features, label))
                qid_mark = qid
