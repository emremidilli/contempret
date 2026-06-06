import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class PatchTokenizer(tf.keras.layers.Layer):
    '''
    A patch tokenizer that reshapes sub-series in patches.
    '''
    def __init__(self, patch_size, nr_of_covariates, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.trainable = False
        self.nr_of_covariates = nr_of_covariates

        self.reshaper = tf.keras.layers.Reshape(
            (-1, patch_size * nr_of_covariates))

    def call(self, x):
        '''
        inputs:
            x: (None, timesteps, covariates)
        returns:
            y: (None, nr_of_patches, covariates x patch_size)
                it will act as (None, timesteps, features) at
                rest of the model.
        '''
        y = self.reshaper(x)

        return y


@tf.keras.saving.register_keras_serializable()
class TrendSeasonalityTokenizer(tf.keras.layers.Layer):
    '''
    A tokenizer that decomposes a time-series into
    trend, seasonality and residual components.
    '''
    def __init__(
            self,
            pool_size_trend,
            nr_of_covariates,
            sigma,
            **kwargs):
        super().__init__(**kwargs)

        self.sigma = sigma

        self.avg_pool_trend = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size_trend,
            strides=1,
            padding='same',
            data_format='channels_last')

        self.trainable = False

        self.nr_of_covariates = nr_of_covariates

    def call(self, x):
        '''
        x: lookback normalized series that is patched
        (None, timesteps, features, covariates)

        for each patch
            reduces the input by avg pooling
            calculates the trend component by avg pooling
            calculates the seasonality componenet by
                subtracting the trend component from
                the original input
                outliers of seasonality component has been replaced
                according to mu +- 3std
            calculates the residual component by subtracting
                trend and seasonality components from the
                original input.

        returns:  tuple of 3 elements
            1. y_trend - (None, timesteps, covariates)
            2. y_seasonality - (None, timesteps, covariates)
            3. y_residual - (None, timesteps, covariates)
        '''

        tres = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        seas = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        reses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in range(0, self.nr_of_covariates):
            to_decompose = x[:, :, i]

            to_decompose = tf.expand_dims(to_decompose, axis=-1)

            tre_to_add = self.avg_pool_trend(to_decompose)

            sea_to_add = tf.subtract(to_decompose, tre_to_add)

            mu = tf.reduce_mean(sea_to_add)
            std = tf.math.reduce_std(sea_to_add)

            ucl = tf.add(mu, tf.multiply(std, self.sigma))
            lcl = tf.add(mu, tf.multiply(std, -self.sigma))

            ucl = tf.zeros_like(sea_to_add) + ucl
            lcl = tf.zeros_like(sea_to_add) + lcl

            sea_to_add = tf.minimum(sea_to_add, ucl)
            sea_to_add = tf.maximum(sea_to_add, lcl)

            res_to_add = tf.subtract(to_decompose, tre_to_add)
            res_to_add = tf.subtract(res_to_add, sea_to_add)

            tre_to_add = tf.squeeze(tre_to_add, axis=-1)
            sea_to_add = tf.squeeze(sea_to_add, axis=-1)
            res_to_add = tf.squeeze(res_to_add, axis=-1)

            tres = tres.write(i, tre_to_add)
            seas = seas.write(i, sea_to_add)
            reses = reses.write(i, res_to_add)

        y_trend = tres.stack()
        y_seasonality = seas.stack()
        y_residual = reses.stack()

        y_trend = tf.transpose(y_trend, perm=[1, 2, 0])
        y_seasonality = tf.transpose(y_seasonality, perm=[1, 2, 0])
        y_residual = tf.transpose(y_residual, perm=[1, 2, 0])

        return (y_trend, y_seasonality, y_residual)
