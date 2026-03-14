from . import PositionEmbedding, TransformerEncoder, Time2Vec

import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class Representation(tf.keras.layers.Layer):
    '''Encoder representation layer.'''
    def __init__(
            self,
            nr_of_encoder_blocks: int,
            nr_of_heads: int,
            dropout_rate: float,
            encoder_ffn_units: int,
            embedding_dims: int,
            use_time2vec: bool,
            prefer_dense_to_time2vec: bool,
            **kwargs):

        super().__init__(**kwargs)

        self.embedding_dims = embedding_dims

        self.pe_tre_temporal = PositionEmbedding(
            embedding_dims=embedding_dims,
            name='pe_tre_temporal')
        self.pe_sea_temporal = PositionEmbedding(
            embedding_dims=embedding_dims,
            name='pe_sea_temporal')
        self.pe_res_temporal = PositionEmbedding(
            embedding_dims=embedding_dims,
            name='pe_res_temporal')

        self.use_time2vec = use_time2vec
        self.prefer_dense_to_time2vec = \
            prefer_dense_to_time2vec
        if self.use_time2vec:
            self.time2vec = Time2Vec(
                embedding_dims=self.embedding_dims,
                prefer_dense_to_time2vec=prefer_dense_to_time2vec,
                name='time2vec')
        else:
            self.time2vec = None

        self.concat_temporals = tf.keras.layers.Concatenate(axis=1)

        self.encoders_temporal = []
        for i in range(nr_of_encoder_blocks):
            self.encoders_temporal.append(
                TransformerEncoder(
                    embed_dim=embedding_dims,
                    num_heads=nr_of_heads,
                    feedforward_dim=encoder_ffn_units,
                    dropout_rate=dropout_rate,
                    name=f'encoders_temporal{i}'
                ))

    def call(self, x):
        '''
        inputs: tuple of 4 elements
            trend: (None, timesteps, features)
            seasonality: (None, timesteps, features)
            residual: (None, timesteps, features)
            dates: (None, features)

        in case "dates" is None, it will not use Time2Vec layer.
        '''

        x_tre_temp, x_sea_temp, x_res_temp, x_dates = x

        x_tre_temp = self.pe_tre_temporal(x_tre_temp)
        x_sea_temp = self.pe_sea_temporal(x_sea_temp)
        x_res_temp = self.pe_res_temporal(x_res_temp)

        if self.use_time2vec:
            x_dates = self.time2vec(x_dates)
            x_temp = self.concat_temporals(
                [x_tre_temp, x_sea_temp, x_res_temp, x_dates])
        else:
            x_temp = self.concat_temporals(
                [x_tre_temp, x_sea_temp, x_res_temp])

        for encoder in self.encoders_temporal:
            x_temp = encoder(x_temp)

        return x_temp
