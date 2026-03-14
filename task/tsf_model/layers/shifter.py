import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class TimeStepShifter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):
        '''
        inputs: tuple of 4 elements
            x_tre: (None, timesteps, features)
            x_sea: (None, timesteps, features)
            x_res: (None, timesteps, features)
            shift: integer

        shifts the timesteps with roll operation.

        outputs: tuple of 3 shifted elements
            y_tre: (None, timesteps, features)
            y_sea: (None, timesteps, features)
            y_res: (None, timesteps, features)
        '''
        x_tre, x_sea, x_res, shift = inputs

        y_tre = tf.roll(x_tre, shift=shift, axis=1)
        y_sea = tf.roll(x_sea, shift=shift, axis=1)
        y_res = tf.roll(x_res, shift=shift, axis=1)

        return (y_tre, y_sea, y_res)
