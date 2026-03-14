from tsf_model.layers import TrendSeasonalityTokenizer

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

        y_lb_tre, y_lb_sea, y_lb_res = self.trend_seasonality_tokenizer(x_lb)

        if self.use_timestamp:
            y_ts = x_ts
        else:
            batch_size = tf.shape(x_lb)[0]
            y_ts = tf.zeros((batch_size, 0), dtype=tf.float32)

        return (y_lb_tre, y_lb_sea, y_lb_res, y_ts)
