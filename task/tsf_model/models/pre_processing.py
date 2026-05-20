from tsf_model.layers import TrendSeasonalityTokenizer

import numpy as np
import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class InputPreProcessor(tf.keras.Model):
    '''
    Preprocessor model for input of pre-training and fine-tuning models.
    The model is not a trainable model.
    That's why, it is not called with .fit() method.
    Instead, it is called with .call() method.
    The input series might be multivariate series.
    It decomposes the input series to
        trend, seasonality and resiudal components for each covariate.
    It is optional to provide timestamp features as an input.
    Timestamp features must be common accross all covariates.
    '''
    def __init__(
            self,
            pool_size_trend: int,
            nr_of_covariates: int,
            sigma: float,
            scale_data: bool,
            use_timestamp: bool,
            **kwargs):
        super().__init__(**kwargs)
        '''
        args:
            pool_size_trend (int): average pool size for trend component.
            nr_of_covariates (int):  number of covariates.
            sigma (float): standard deviation to calculate residuals.
            scale_data (bool): to scale dataset or not.
            use_timestamp (bool): to use timestamp features or not.
        '''

        self.scale_data = scale_data
        self.use_timestamp = use_timestamp

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            pool_size_trend=pool_size_trend,
            nr_of_covariates=nr_of_covariates,
            sigma=sigma)

    # cuDNN pooling fails on very large batch sizes (e.g. ETTm ~240k samples)
    _CHUNK_SIZE = 10000

    def call(self, inputs):
        '''
        inputs: tuple of 2 elements.
        Each element is a tf.data.Dataset object.
            x_lb: (None, timesteps, covariates)
            x_ts: (None, features) are timestamp features.
                It must be provided if use_timestamp is True.

        returns tuple of 4 elemements.
            y_lb_tre: (None, timesteps, covariates)
            y_lb_sea: (None, timesteps, covariates)
            y_lb_res: (None, timesteps, covariates)
            y_lb_ts: (None, features) optional if it timestamp features
                are provided.
        '''
        if self.use_timestamp:
            (x_lb, x_ts) = inputs
            x_ts = tf.cast(x_ts, dtype=tf.float32)
        else:
            (x_lb, ) = inputs
            x_ts = None

        n = x_lb.shape[0]
        if n is not None and n > self._CHUNK_SIZE:
            chunks_tre, chunks_sea, chunks_res = [], [], []
            for start in range(0, n, self._CHUNK_SIZE):
                chunk = x_lb[start:start + self._CHUNK_SIZE]
                tre, sea, res = self.trend_seasonality_tokenizer(chunk)
                chunks_tre.append(tre.numpy())
                chunks_sea.append(sea.numpy())
                chunks_res.append(res.numpy())
            with tf.device('/CPU:0'):
                y_lb_tre = tf.constant(np.concatenate(chunks_tre, axis=0))
                y_lb_sea = tf.constant(np.concatenate(chunks_sea, axis=0))
                y_lb_res = tf.constant(np.concatenate(chunks_res, axis=0))
        else:
            y_lb_tre, y_lb_sea, y_lb_res = \
                self.trend_seasonality_tokenizer(x_lb)

        if self.use_timestamp:
            y_ts = x_ts
        else:
            batch_size = tf.shape(x_lb)[0]
            y_ts = tf.zeros((batch_size, 0), dtype=tf.float32)

        return (y_lb_tre, y_lb_sea, y_lb_res, y_ts)
