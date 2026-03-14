from . import CausalSelfAttention, CrossAttention, FeedForward

import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class LinearHead(tf.keras.layers.Layer):
    '''Decoder for masked auto-encoder.'''
    def __init__(self, nr_of_timesteps, nr_of_covariates, **kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=nr_of_timesteps * nr_of_covariates,
            use_bias=True)

        self.reshaper = tf.keras.layers.Reshape(
            target_shape=(nr_of_timesteps, nr_of_covariates))

    def call(self, x):
        '''
        args:
            x: (None, timesteps, covariates)

        returns:
            y: (None, timesteps, covariates)
        '''
        x = self.flatten(x)

        y = self.dense(x)

        y = self.reshaper(y)

        return y


@tf.keras.saving.register_keras_serializable()
class ProjectionHead(tf.keras.layers.Layer):
    '''Projection head for contrastive learning task.'''
    def __init__(self, iFfnUnits, **kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=iFfnUnits,
            activation='relu',
            use_bias=False)

        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, feature)
        '''

        x = self.flatten(x)

        y = self.dense(x)

        y = self.layer_norm(y)

        return y


@tf.keras.saving.register_keras_serializable()
class DecoderBlock(tf.keras.layers.Layer):
    '''Transformer decoder block.'''
    def __init__(
            self,
            *,
            hidden_dims,
            nr_of_heads,
            dropout_rate,
            dff,
            **kwargs):
        super().__init__(**kwargs)

        self.hidden_dims = hidden_dims
        self.nr_of_heads = nr_of_heads
        self.dropout_rate = dropout_rate

        self.causal_self_attn = CausalSelfAttention(
            num_heads=nr_of_heads,
            key_dim=hidden_dims)

        self.cross_attn = CrossAttention(
            num_heads=nr_of_heads,
            key_dim=hidden_dims)

        self.ffn = FeedForward(
            d_model=hidden_dims,
            dff=dff,
            dropout_rate=dropout_rate)

    def call(self, inputs):
        '''
        x: (None, time_steps, features)
        context: (None, time_steps, features)
        y: (None, time_steps, features)
        '''
        x, context = inputs
        x = self.causal_self_attn(x=x)
        x = self.cross_attn(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attn.last_attn_scores

        y = self.ffn(x)
        return y


@tf.keras.saving.register_keras_serializable()
class TransformerDecoder(tf.keras.layers.Layer):
    '''Decoder for singe-step predictor.'''
    def __init__(
            self,
            *,
            num_layers,
            hidden_dims,
            nr_of_heads,
            dff,
            dropout_rate,
            **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.nr_of_heads = nr_of_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderBlock(
                hidden_dims=hidden_dims,
                nr_of_heads=nr_of_heads,
                dff=dff,
                dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, inputs):
        '''
        x: (None, time_steps, features)
        y: (None, time_steps, features)
        '''
        x, context = inputs
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i]((x, context))

        y = x
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return y
