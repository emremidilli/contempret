
import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):
    '''Base attention layer that can be inhereted.'''
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


@tf.keras.saving.register_keras_serializable()
class CausalSelfAttention(BaseAttention):
    '''Causal self attention where causal mask is applied.'''
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@tf.keras.saving.register_keras_serializable()
class CrossAttention(BaseAttention):
    '''Cross attention layer where key and values are from context.'''
    def call(self, x, context):
        '''
        x: (None, time_steps, features)
        context: (None, time_steps, features)
        y: (None, time_steps, features)
        '''
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        y = self.layernorm(x)

        return y


@tf.keras.saving.register_keras_serializable()
class GlobalSelfAttention(BaseAttention):
    '''A self-attention mechanism that can be part of transformer encoder \
        and/or decoder.'''
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
