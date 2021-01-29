import collections


class SparseFeature(
    collections.namedtuple(
        "SparseFeature",
        ["feature_name", "vocab_size", "embed_dim", "dtype", "mask_zero"])):
    """
    Fields:
        feature_name: the name of sparse feature
        vocab_size: the size of spare feature vocabulary
        embed_size: the embedding dimension size of feature
        mask_zero: Whether the index 0 is used for padding
    """

    def __new__(cls, feature_name, vocab_size, embed_size, dtype="int32", mask_zero=False):
        return super(SparseFeature, cls).__new__(
            cls, feature_name, vocab_size, embed_size, dtype, mask_zero)


class DenseFeature(
    collections.namedtuple(
        "DenseFeature",
        ["feature_name", "dtype"])):
    def __new__(cls, feature_name, dtype='float32'):
        return super(DenseFeature, cls).__new__(
            cls, feature_name, dtype)


class SequenceFeature(
    collections.namedtuple(
        "SequenceFeature",
        ["feature_name", "seq_len", "element_sparse_feature"])):
    """
    Fields:
        feature_name: the name of variable sequence feature
        seq_len: the length of sequence
        element_sparse_feature: the element sparse feature of sequence
    """

    def __new__(cls, feature_name, seq_len, element_sparse_feature):
        return super(SequenceFeature, cls).__new__(
            cls, feature_name, seq_len, element_sparse_feature)
