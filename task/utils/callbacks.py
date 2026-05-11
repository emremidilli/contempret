import gc

import tensorflow as tf


class LearningRateCallback(tf.keras.callbacks.Callback):
    '''Noam Learning rate schedule of "Attention is all you need paper"'''
    def __init__(
            self,
            d_model,
            warmup_steps=4000,
            scale_factor=1.0,
            remained_step_nr=0):
        super().__init__()

        self.d_model = tf.cast(d_model, dtype=tf.float32)

        self.warmup_steps = warmup_steps

        self.scale_factor = tf.cast(scale_factor, dtype=tf.float32)

        self.step_nr = remained_step_nr

    def schedule(self, step):
        '''calculates the new learning rate based on step (batch) number'''
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        unscaled = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        scaled = self.scale_factor * unscaled
        return scaled

    def on_batch_begin(self, batch, logs=None):
        '''sets the calculated learning rate to the optimzer'''
        self.step_nr = self.step_nr + 1
        lr = self.schedule(step=self.step_nr)
        tf.keras.backend.set_value(self.model.mae_comp_optimizer.lr, lr)
        tf.keras.backend.set_value(self.model.mae_tre_optimizer.lr, lr)
        tf.keras.backend.set_value(self.model.mae_sea_optimizer.lr, lr)
        tf.keras.backend.set_value(self.model.cl_optimizer.lr, lr)


class RamCleaner(tf.keras.callbacks.Callback):
    '''Callback to clean RAM with garbage collector.'''

    def on_epoch_end(self, epoch, logs={}):
        '''Cleans the RAM after every epoch.'''
        gc.collect()
