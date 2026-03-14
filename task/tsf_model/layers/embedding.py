import numpy as np

import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class PositionEmbedding(tf.keras.layers.Layer):
    '''Positional embedding from "Attention is all you need" paper.'''
    def __init__(self, embedding_dims, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims

        self.embedding = tf.keras.layers.Dense(
            units=embedding_dims,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')

    def positional_encoding(self, length, depth):
        import numpy as np

        depth = depth / 2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        '''
        input: (None, timesteps, features)
        output: (None, timesteps, features)
        '''
        pos_encodings = self.positional_encoding(
            length=inputs.shape[1],
            depth=self.embedding_dims)

        y = self.embedding(inputs)

        return y + pos_encodings


@tf.keras.saving.register_keras_serializable()
class Time2Vec(tf.keras.layers.Layer):
    '''
        Embedds a datetime vector via perioid activation based on
            "Time2Vec: Learning a Vector Representation of Time" paper.
        As difference from original paper:
            In order to prevent from gradient explosion for higher dimensions,
            layer normalization is employed at end of the layer.
    '''
    def __init__(self, embedding_dims, prefer_dense_to_time2vec, **kwargs):
        super(Time2Vec, self).__init__(**kwargs)

        self.embedding_dims = embedding_dims
        self.prefer_dense_to_time2vec = prefer_dense_to_time2vec

        if self.prefer_dense_to_time2vec:
            self.dense_all = tf.keras.layers.Dense(units=embedding_dims)
        else:
            # Linear term initializers
            linear_kernel_init = tf.keras.initializers.RandomUniform(
                minval=-0.5,
                maxval=0.5)
            linear_bias_init   = tf.keras.initializers.Zeros()

            # Periodic term initializers
            PI = float(np.pi)

            periodic_kernel_init = tf.keras.initializers.RandomUniform(
                minval=-PI,
                maxval=PI)
            periodic_bias_init = tf.keras.initializers.RandomUniform(
                minval=-PI,
                maxval=PI)

            self.dense_linear = tf.keras.layers.Dense(
                units=1,
                kernel_initializer=linear_kernel_init,
                bias_initializer=linear_bias_init)

            self.dense_periodic = tf.keras.layers.Dense(
                units=embedding_dims - 1,
                kernel_initializer=periodic_kernel_init,
                bias_initializer=periodic_bias_init)

            self.concatter = tf.keras.layers.Concatenate(axis=2)

            self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        '''
        inputs: (None, feature_size)
        returns: (None, feature_size, embedding_dims)
        '''

        if self.prefer_dense_to_time2vec:
            x = tf.expand_dims(inputs, axis=1)
            y = self.dense_all(x)
        else:
            x = tf.expand_dims(inputs, axis=2)
            linear = self.dense_linear(x)

            periodic = self.dense_periodic(x)
            periodic = tf.keras.backend.sin(periodic)

            embedded = self.concatter([linear, periodic])

            y = self.layer_norm(embedded)

        return y
