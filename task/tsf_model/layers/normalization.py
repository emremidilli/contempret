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
            name='gamma')
        self.beta = self.add_weight(
            shape=(nr_of_covariates, ),
            initializer='zeros',
            trainable=True,
            name='beta')

    def call(self, inputs):
        '''
        inputs:
            x: (None, timesteps, covariates)

        returns:
            y: (None, timesteps, covariates)
        '''
        x = inputs

        mu = tf.math.reduce_mean(x, axis=1)
        var = tf.math.reduce_variance(x, axis=1)
        var_adj = tf.math.add(var, self.epsilon)
        sigma = tf.math.sqrt(var_adj)

        z = tf.subtract(x, tf.expand_dims(mu, axis=1))
        r = tf.divide(z, tf.expand_dims(sigma, axis=1))

        y = tf.math.multiply(r, self.gamma) + self.beta

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

        mu = tf.math.reduce_mean(x, axis=1)
        var = tf.math.reduce_variance(x, axis=1)
        var_adj = tf.math.add(var, self.epsilon)
        sigma = tf.math.sqrt(var_adj)

        k = tf.math.subtract(y_pred, self.beta)

        m = tf.math.divide(k, self.gamma)

        n = tf.math.multiply(tf.expand_dims(sigma, axis=1), m)

        y = tf.math.add(n, tf.expand_dims(mu, axis=1))

        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            'nr_of_covariates': self.nr_of_covariates,
            'epsilon': self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
