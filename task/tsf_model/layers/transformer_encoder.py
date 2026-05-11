import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class TransformerEncoder(tf.keras.layers.Layer):
    '''
    Single encoder block from "Attention is all you need paper."
    '''
    def __init__(self,
                 embed_dim,
                 num_heads,
                 feedforward_dim,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate

        key_dims = embed_dim // num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dims)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='encoder_norm_1')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(feedforward_dim, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='encoder_norm_2')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        ff_output = self.feedforward(out1)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.norm2(out1 + ff_output)

        return out2
