import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class ReversibleInstanceNormalization(tf.keras.layers.Layer):
    '''
    Reversible Instance Normalizer (RevIN) from the original paper of
        REVERSIBLE INSTANCE NORMALIZATION FOR
        ACCURATE TIME-SERIES FORECASTING AGAINST DISTRIBUTION SHIFT
        (Kim et. al. 2022)
    '''

    def __init__(self, nr_of_covariates, epsilon, **kwargs):
        '''
        In the original paper, beta and gamma variables have
            the same shape with the number of covariates
            in multivariate forecasting model.

        args:
            nr_of_covariates (int): number of covaraites
            epsilon (float): a constant that helps to escape zero division
                errors.
        '''
        super().__init__(**kwargs)

        self.nr_of_covariates = nr_of_covariates
        self.epsilon = epsilon

        self.gamma = self.add_weight(
            shape=(nr_of_covariates, ),
            initializer='ones',
            trainable=True,
            dtype=self.dtype_policy.compute_dtype,
            name='gamma')
        self.beta = self.add_weight(
            shape=(nr_of_covariates, ),
            initializer='zeros',
            trainable=True,
            dtype=self.dtype_policy.compute_dtype,
            name='beta')

    def call(self, inputs):
        '''
        inputs:
            x: (None, timesteps, covariates)

        returns:
            y: (None, timesteps, covariates)
        '''
        x = inputs

        # Cast epsilon to input dtype
        epsilon = tf.cast(self.epsilon, x.dtype)

        mu = tf.math.reduce_mean(x, axis=1)
        var = tf.math.reduce_variance(x, axis=1)
        var_adj = tf.math.add(var, self.epsilon)
        sigma = tf.math.sqrt(var_adj)

        z = tf.subtract(x, tf.expand_dims(mu, axis=1))
        r = tf.divide(z, tf.expand_dims(sigma, axis=1))

        gamma = tf.cast(self.gamma, x.dtype)
        beta = tf.cast(self.beta, x.dtype)

        y = tf.math.multiply(r, gamma) + beta

        return y

    def denormalize(self, inputs):
        '''
        inputs:
            x: (None, timesteps, covariates)
            y_pred: (None, timesteps, covariates)
                scaled prediction

        returns:
            y_act: (None, timesteps, covariates)
                final prediction
        '''
        x, y_pred = inputs

        # Cast epsilon to input dtype
        target_dtype = y_pred.dtype
        x = tf.cast(x, target_dtype)
        epsilon = tf.cast(self.epsilon, target_dtype)

        mu = tf.math.reduce_mean(x, axis=1)
        var = tf.math.reduce_variance(x, axis=1)
        var_adj = tf.math.add(var, self.epsilon)
        sigma = tf.math.sqrt(var_adj)

        gamma = tf.cast(self.gamma, target_dtype)
        beta = tf.cast(self.beta, target_dtype)

        k = tf.math.subtract(y_pred, beta)

        m = tf.math.divide(k, gamma)

        n = tf.math.multiply(tf.expand_dims(sigma, axis=1), m)

        y = tf.math.add(n, tf.expand_dims(mu, axis=1))

        return y
