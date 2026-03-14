import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    '''Feed-forward layer of transformer archicture.'''
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='gelu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        '''
        x: (None, time_steps, features)
        y: (None, time_steps, features)
        '''
        x = self.add([x, self.seq(x)])
        y = self.layer_norm(x)
        return y
