import gc
import time

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
        if hasattr(self.model, 'mae_comp_optimizer'):
            tf.keras.backend.set_value(self.model.mae_comp_optimizer.lr, lr)

        if hasattr(self.model, 'mae_tre_optimizer'):
            tf.keras.backend.set_value(self.model.mae_tre_optimizer.lr, lr)

        if hasattr(self.model, 'mae_sea_optimizer'):
            tf.keras.backend.set_value(self.model.mae_sea_optimizer.lr, lr)

        if hasattr(self.model, 'mae_sea_optimizer'):
            tf.keras.backend.set_value(self.model.cl_optimizer.lr, lr)

        if hasattr(self.model, 'optimizer'):
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class RamCleaner(tf.keras.callbacks.Callback):
    '''Callback to clean RAM with garbage collector.'''

    def on_epoch_end(self, epoch, logs={}):
        '''Cleans the RAM after every epoch.'''
        gc.collect()


class TimingCallback(tf.keras.callbacks.Callback):
    '''Records epoch time, mean step time, and step count into history.'''

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()
        self._n_steps = 0

    def on_train_batch_end(self, batch, logs=None):
        self._n_steps += 1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time_s = time.perf_counter() - self._epoch_start
        logs['epoch_time_s'] = epoch_time_s
        logs['n_steps'] = self._n_steps
        logs['step_time_ms'] = (epoch_time_s * 1000 / self._n_steps) if self._n_steps else 0.0
