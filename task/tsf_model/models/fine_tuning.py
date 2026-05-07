import tensorflow as tf

from tsf_model.layers import LinearHead, \
    ReversibleInstanceNormalization


@tf.keras.saving.register_keras_serializable()
class SequencePredictor(tf.keras.Model):
    '''Keras model for fine-tuning univariate time series.'''
    def __init__(
            self,
            patch_tokenizer,
            encoder_representation,
            nr_of_timesteps,
            trend_prompt,
            seasonality_prompt,
            residual_prompt,
            revIn_tre,
            revIn_sea,
            revIn_res,
            tre_embedding,
            sea_embedding,
            res_embedding,
            tune_time2vec,
            **kwargs):
        '''
        args:

        '''
        super().__init__(**kwargs)

        self.embedding_dims = \
            encoder_representation.get_config()['embedding_dims']

        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res

        self.tre_embedding = tre_embedding
        self.sea_embedding = sea_embedding
        self.res_embedding = res_embedding

        self.decoder_tre = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='lienar_head_trend')
        self.decoder_sea = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='lienar_head_seasonality')
        self.decoder_res = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='lienar_head_residual')

        self.patch_tokenizer = patch_tokenizer
        self.encoder_representation = encoder_representation
        self.trend_prompt = trend_prompt
        self.seasonality_prompt = seasonality_prompt
        self.residual_prompt = residual_prompt

        self.nr_of_timesteps = nr_of_timesteps

        self.timesteps_concatter = tf.keras.layers.Concatenate(axis=1)
        self.tune_time2vec = tune_time2vec

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

        tre_embed = self.tre_embedding(tre_patch)
        sea_embed = self.sea_embedding(sea_patch)
        res_embed = self.res_embedding(res_patch)

        if self.trend_prompt is not None:
            tre_prompts = self.trend_prompt(tre_norm)
            sea_prompts = self.seasonality_prompt(sea_norm)
            res_prompts = self.residual_prompt(res_norm)

            tre_embed = self.timesteps_concatter([tre_prompts, tre_embed])
            sea_embed = self.timesteps_concatter([sea_prompts, sea_embed])
            res_embed = self.timesteps_concatter([res_prompts, res_embed])

        y_cont_temp = self.encoder_representation(
            (tre_embed, sea_embed, res_embed, dates))

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'nr_of_timesteps': self.nr_of_timesteps,
                'tune_time2vec': self.tune_time2vec,
                'revIn_tre': tf.keras.layers.serialize(self.revIn_tre),
                'revIn_sea': tf.keras.layers.serialize(self.revIn_sea),
                'revIn_res': tf.keras.layers.serialize(self.revIn_res),
                'tre_embedding': tf.keras.layers.serialize(self.tre_embedding),
                'sea_embedding': tf.keras.layers.serialize(self.sea_embedding),
                'res_embedding': tf.keras.layers.serialize(self.res_embedding),
                'patch_tokenizer': tf.keras.layers.serialize(
                    self.patch_tokenizer),
                'encoder_representation': tf.keras.layers.serialize(
                    self.encoder_representation),
                'trend_prompt': tf.keras.layers.serialize(self.trend_prompt),
                'seasonality_prompt': tf.keras.layers.serialize(self.seasonality_prompt),
                'residual_prompt': tf.keras.layers.serialize(self.residual_prompt),
                'decoder_tre': tf.keras.layers.serialize(self.decoder_tre),
                'decoder_sea': tf.keras.layers.serialize(self.decoder_sea),
                'decoder_res': tf.keras.layers.serialize(self.decoder_res)
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config['revIn_tre'] = tf.keras.layers.deserialize(
            config['revIn_tre'])
        config['revIn_sea'] = tf.keras.layers.deserialize(
            config['revIn_sea'])
        config['revIn_res'] = tf.keras.layers.deserialize(
            config['revIn_res'])
        config['tre_embedding'] = tf.keras.layers.deserialize(
            config['tre_embedding'])
        config['sea_embedding'] = tf.keras.layers.deserialize(
            config['sea_embedding'])
        config['res_embedding'] = tf.keras.layers.deserialize(
            config['res_embedding'])
        config['patch_tokenizer'] = tf.keras.layers.deserialize(
            config['patch_tokenizer'])
        config['encoder_representation'] = tf.keras.layers.deserialize(
            config['encoder_representation'])
        config['trend_prompt'] = tf.keras.layers.deserialize(
            config['trend_prompt'])
        config['seasonality_prompt'] = tf.keras.layers.deserialize(
            config['seasonality_prompt'])
        config['residual_prompt'] = tf.keras.layers.deserialize(
            config['residual_prompt'])
        config['decoder_tre'] = tf.keras.layers.deserialize(
            config['decoder_tre'])
        config['decoder_sea'] = tf.keras.layers.deserialize(
            config['decoder_sea'])
        config['decoder_res'] = tf.keras.layers.deserialize(
            config['decoder_res'])

        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class TimeSeriesClassifier(tf.keras.Model):
    '''Keras model for fine-tuning univariate time series.'''
    def __init__(
            self,
            patch_tokenizer,
            encoder_representation,
            nr_of_timesteps,
            trend_prompt,
            seasonality_prompt,
            residual_prompt,
            revIn_tre,
            revIn_sea,
            revIn_res,
            tre_embedding,
            sea_embedding,
            res_embedding,
            tune_time2vec,
            **kwargs):
        '''
        args:

        '''
        super().__init__(**kwargs)

        self.embedding_dims = \
            encoder_representation.get_config()['embedding_dims']

        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res

        self.tre_embedding = tre_embedding
        self.sea_embedding = sea_embedding
        self.res_embedding = res_embedding

        self.patch_tokenizer = patch_tokenizer
        self.encoder_representation = encoder_representation
        self.trend_prompt = trend_prompt
        self.seasonality_prompt = seasonality_prompt
        self.residual_prompt = residual_prompt

        self.linear_head = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='lienar_head')

        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.nr_of_timesteps = nr_of_timesteps

        self.timesteps_concatter = tf.keras.layers.Concatenate(axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'revIn_tre': tf.keras.layers.serialize(self.revIn_tre),
                'revIn_sea': tf.keras.layers.serialize(self.revIn_sea),
                'revIn_res': tf.keras.layers.serialize(self.revIn_res),
                'tre_embedding': tf.keras.layers.serialize(self.tre_embedding),
                'sea_embedding': tf.keras.layers.serialize(self.sea_embedding),
                'res_embedding': tf.keras.layers.serialize(self.res_embedding),
                'patch_tokenizer': tf.keras.layers.serialize(
                    self.patch_tokenizer),
                'encoder_representation': tf.keras.layers.serialize(
                    self.encoder_representation),
                'trend_prompt': tf.keras.layers.serialize(self.trend_prompt),
                'seasonality_prompt': tf.keras.layers.serialize(self.seasonality_prompt),
                'residual_prompt': tf.keras.layers.serialize(self.residual_prompt)
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config['revIn_tre'] = tf.keras.layers.deserialize(
            config['revIn_tre'])
        config['revIn_sea'] = tf.keras.layers.deserialize(
            config['revIn_sea'])
        config['revIn_res'] = tf.keras.layers.deserialize(
            config['revIn_res'])
        config['tre_embedding'] = tf.keras.layers.deserialize(
            config['tre_embedding'])
        config['sea_embedding'] = tf.keras.layers.deserialize(
            config['sea_embedding'])
        config['res_embedding'] = tf.keras.layers.deserialize(
            config['res_embedding'])
        config['patch_tokenizer'] = tf.keras.layers.deserialize(
            config['patch_tokenizer'])
        config['encoder_representation'] = tf.keras.layers.deserialize(
            config['encoder_representation'])
        config['trend_prompt'] = tf.keras.layers.deserialize(
            config['trend_prompt'])
        config['seasonality_prompt'] = tf.keras.layers.deserialize(
            config['seasonality_prompt'])
        config['residual_prompt'] = tf.keras.layers.deserialize(
            config['residual_prompt'])

        return cls(**config)

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

        tre_embed = self.tre_embedding(tre_patch)
        sea_embed = self.sea_embedding(sea_patch)
        res_embed = self.res_embedding(res_patch)

        if self.trend_prompt is not None:
            tre_prompts = self.trend_prompt(tre_norm)
            sea_prompts = self.seasonality_prompt(sea_norm)
            res_prompts = self.residual_prompt(res_norm)

            tre_embed = self.timesteps_concatter([tre_prompts, tre_embed])
            sea_embed = self.timesteps_concatter([sea_prompts, sea_embed])
            res_embed = self.timesteps_concatter([res_prompts, res_embed])

        y_cont_temp = self.encoder_representation(
            (tre_embed, sea_embed, res_embed, dates))

        logits = self.linear_head(y_cont_temp)

        logits = self.sigmoid(logits)

        return logits
