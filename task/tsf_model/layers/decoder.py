
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

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            dtype=tf.float32,
            name='projection_head_layer_norm')

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, feature)
        '''

        x = self.flatten(x)

        y = self.dense(x)

        y = self.layer_norm(y)

        return y
