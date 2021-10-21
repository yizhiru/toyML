import tensorflow as tf
from official.nlp import keras_nlp
from tensorflow.python.keras.engine.base_layer import Layer


class TransformerEncoder(Layer):
    """Encoder of Transformer."""

    def __init__(self,
                 num_blocks,
                 num_heads,
                 ffn_dim,
                 attention_dropout=0.0,
                 mha_output_dropout=0.1,
                 ffn_output_dropout=0.1,
                 norm_epsilon=1e-12,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self._num_blocks = num_blocks
        self._num_heads = num_heads
        self._ffn_dim = ffn_dim
        self._attention_dropout = attention_dropout
        self._mha_output_dropout = mha_output_dropout
        self._ffn_output_dropout = ffn_output_dropout
        self._norm_epsilon = norm_epsilon

        self.supports_masking = True

        self._pos_embed = keras_nlp.layers.PositionEmbedding(max_length=20)
        self._encoder_blocks = [
            keras_nlp.layers.TransformerEncoderBlock(num_attention_heads=self._num_heads,
                                                     inner_dim=self._ffn_dim,
                                                     inner_activation='relu')
            for _ in range(self._num_blocks)]

    def compute_mask(self, inputs, mask=None):
        return mask

    @staticmethod
    def _padding_attention_mask(key_mask):
        """
        compute attention mask that prevents attention to certain positions.

        Args:
          key_mask: shape of [B, Sk]
        """
        mask = tf.expand_dims(key_mask, axis=1)
        key_size = tf.shape(key_mask)[-1]
        mask = tf.gather(mask, tf.zeros([key_size], tf.int32), axis=1)
        return tf.cast(mask, dtype=tf.bool)

    def call(self, inputs, mask=None, training=None):
        pos_embed = self._pos_embed(inputs)
        output = tf.add(inputs, pos_embed)
        attention_mask = self._padding_attention_mask(mask)
        for layer in self._encoder_blocks:
            output = layer([output, attention_mask], training=training)

        return output
