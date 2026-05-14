import tensorflow as tf

from tsf_model.layers import LinearHead, \
    ReversibleInstanceNormalization


@tf.keras.saving.register_keras_serializable()
class BaseFineTuning(tf.keras.Model):
    def __init__(
        self,
        revIn_tre,
        revIn_sea,
        revIn_res,
        patch_tokenizer,
        representation,
        nr_of_timesteps,
        tune_time2vec,
        **kwargs):
        '''
        args:

        '''
        super().__init__(**kwargs)

        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res
        self.patch_tokenizer = patch_tokenizer
        self.representation = representation
        self.nr_of_timesteps = nr_of_timesteps
        self.tune_time2vec = tune_time2vec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'revIn_tre': tf.keras.layers.serialize(self.revIn_tre),
                'revIn_sea': tf.keras.layers.serialize(self.revIn_sea),
                'revIn_res': tf.keras.layers.serialize(self.revIn_res),
                'patch_tokenizer': tf.keras.layers.serialize(self.patch_tokenizer),
                'representation': tf.keras.layers.serialize(self.representation),
                'nr_of_timesteps': self.nr_of_timesteps,
                'tune_time2vec': self.tune_time2vec,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config['revIn_tre'] = tf.keras.layers.deserialize(config['revIn_tre'])
        config['revIn_sea'] = tf.keras.layers.deserialize(config['revIn_sea'])
        config['revIn_res'] = tf.keras.layers.deserialize(config['revIn_res'])
        config['patch_tokenizer'] = tf.keras.layers.deserialize(config['patch_tokenizer'])
        config['representation'] = tf.keras.layers.deserialize(config['representation'])

        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class SequencePredictor(BaseFineTuning):
    '''Keras model for fine-tuning univariate time series.'''
    def __init__(
        self,
        revIn_tre,
        revIn_sea,
        revIn_res,
        patch_tokenizer,
        representation,
        nr_of_timesteps,
        tune_time2vec,
        **kwargs):
        '''
        args:

        '''
        super().__init__(
            revIn_tre=revIn_tre,
            revIn_sea=revIn_sea,
            revIn_res=revIn_res,
            patch_tokenizer=patch_tokenizer,
            representation=representation,
            nr_of_timesteps=nr_of_timesteps,
            tune_time2vec=tune_time2vec,
            **kwargs)

        self.decoder_tre = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='linear_head_trend')
        self.decoder_sea = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='linear_head_seasonality')
        self.decoder_res = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='linear_head_residual')

    def call(self, inputs):
        '''
        Timesteps of forecast horizon are masked.
        args:
            tre: (None, timesteps, 1)
            sea: (None, timesteps, 1)
            res: (None, timesteps, 1)
            dates: (None, features)
        returns:
            pred: (None, timesteps, 1)
        '''
        tre, sea, res, dates = inputs

        # instance normalize
        tre_norm = self.revIn_tre(tre)
        sea_norm = self.revIn_sea(sea)
        res_norm = self.revIn_res(res)

        # tokenize timesteps into patches
        tre_patch = self.patch_tokenizer(tre_norm)
        sea_patch = self.patch_tokenizer(sea_norm)
        res_patch = self.patch_tokenizer(res_norm)

        y_cont_temp = self.representation((tre_patch, sea_patch, res_patch, dates))

        y_pred_tre = self.decoder_tre(y_cont_temp)
        y_pred_sea = self.decoder_sea(y_cont_temp)
        y_pred_res = self.decoder_res(y_cont_temp)

        # instance denormalize
        y_pred_tre = self.revIn_tre.denormalize((tre, y_pred_tre))
        y_pred_sea = self.revIn_sea.denormalize((sea, y_pred_sea))
        y_pred_res = self.revIn_res.denormalize((res, y_pred_res))

        # compose
        pred = y_pred_tre + y_pred_sea + y_pred_res

        return pred


@tf.keras.saving.register_keras_serializable()
class TimeSeriesClassifier(BaseFineTuning):
    '''Keras model for fine-tuning univariate time series.'''
    def __init__(
        self,
        revIn_tre,
        revIn_sea,
        revIn_res,
        patch_tokenizer,
        representation,
        nr_of_timesteps,
        tune_time2vec,
        **kwargs):
        '''
        args:

        '''
        super().__init__(
            revIn_tre=revIn_tre,
            revIn_sea=revIn_sea,
            revIn_res=revIn_res,
            patch_tokenizer=patch_tokenizer,
            representation=representation,
            nr_of_timesteps=nr_of_timesteps,
            tune_time2vec=tune_time2vec,
            **kwargs)

        self.linear_head = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='linear_head')

        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        '''
        Timesteps of forecast horizon are masked.
        args:
            tre: (None, timesteps, 1)
            sea: (None, timesteps, 1)
            res: (None, timesteps, 1)
            dates: (None, features)
        returns:
            pred: (None, timesteps, 1)
        '''
        tre, sea, res, dates = inputs

        # instance normalize
        tre_norm = self.revIn_tre(tre)
        sea_norm = self.revIn_sea(sea)
        res_norm = self.revIn_res(res)

        # tokenize timesteps into patches
        tre_patch = self.patch_tokenizer(tre_norm)
        sea_patch = self.patch_tokenizer(sea_norm)
        res_patch = self.patch_tokenizer(res_norm)

        y_cont_temp = self.representation((tre_patch, sea_patch, res_patch, dates))

        logits = self.linear_head(y_cont_temp)

        logits = self.sigmoid(logits)

        return logits
