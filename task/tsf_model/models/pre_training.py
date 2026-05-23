import numpy as np

import tensorflow as tf
from typing import Callable, Optional, Tuple, Union

from tsf_model.layers import Representation, \
    LinearHead, ProjectionHead, PatchMasker, TimeStepShifter, \
    ReversibleInstanceNormalization, PatchTokenizer, \
    SoftPrompts

from utils import (
    LearningRateCallback,
    RamCleaner,
    TimingCallback)


@tf.keras.saving.register_keras_serializable()
class RepresentationLearning(tf.keras.Model):
    def __init__(
        self,
        nr_of_covariates: int,
        nr_of_patches: int,
        nr_of_encoder_blocks: int,
        nr_of_heads: int,
        dropout_rate: float,
        encoder_ffn_units: int,
        embedding_dims: int,
        nr_of_timesteps: int,
        prompt_pool_size: int,
        nr_of_most_similar_prompts: int,
        use_time2vec: bool,
        l1_trend: float,
        l2_trend: float,
        l1_seasonality: float,
        l2_seasonality: float,
        l1_residual: float,
        l2_residual: float,
        prefer_dense_to_time2vec: bool,
        custom_prompt_keys_trend: list,
        custom_prompt_keys_seasonality: list,
        custom_prompt_keys_residual: list,
        **kwargs):
        super().__init__(**kwargs)

        self.nr_of_covariates = nr_of_covariates
        self.nr_of_patches = nr_of_patches
        self.nr_of_encoder_blocks = nr_of_encoder_blocks
        self.nr_of_heads = nr_of_heads
        self.dropout_rate = dropout_rate
        self.encoder_ffn_units = encoder_ffn_units
        self.embedding_dims = embedding_dims
        self.nr_of_timesteps = nr_of_timesteps
        self.prompt_pool_size = prompt_pool_size
        self.nr_of_most_similar_prompts = nr_of_most_similar_prompts
        self.use_time2vec = use_time2vec
        self.l1_trend = l1_trend
        self.l2_trend = l2_trend
        self.l1_seasonality = l1_seasonality
        self.l2_seasonality = l2_seasonality
        self.l1_residual = l1_residual
        self.l2_residual = l2_residual
        self.prefer_dense_to_time2vec = prefer_dense_to_time2vec
        self.custom_prompt_keys_trend = custom_prompt_keys_trend
        self.custom_prompt_keys_seasonality = custom_prompt_keys_seasonality
        self.custom_prompt_keys_residual = custom_prompt_keys_residual

        self.trend_prompt = None
        self.seasonality_prompt = None
        self.residual_prompt = None
        if self.prompt_pool_size > 0:
            self.trend_prompt = SoftPrompts(
                key_dims=self.nr_of_timesteps * self.nr_of_covariates,
                embedding_dims=self.embedding_dims,
                nr_of_query_patches=self.nr_of_patches,
                prompt_pool_size=self.prompt_pool_size,
                nr_of_most_similar_prompts=self.nr_of_most_similar_prompts,
                custom_prompt_keys=self.custom_prompt_keys_trend)

            self.seasonality_prompt = SoftPrompts(
                key_dims=self.nr_of_timesteps * self.nr_of_covariates,
                embedding_dims=self.embedding_dims,
                nr_of_query_patches=self.nr_of_patches,
                prompt_pool_size=self.prompt_pool_size,
                nr_of_most_similar_prompts=self.nr_of_most_similar_prompts,
                custom_prompt_keys=self.custom_prompt_keys_seasonality)

            self.residual_prompt = SoftPrompts(
                key_dims=self.nr_of_timesteps * self.nr_of_covariates,
                embedding_dims=self.embedding_dims,
                nr_of_query_patches=self.nr_of_patches,
                prompt_pool_size=self.prompt_pool_size,
                nr_of_most_similar_prompts=self.nr_of_most_similar_prompts,
                custom_prompt_keys=self.custom_prompt_keys_residual)

        self.tre_embedding = tf.keras.layers.Dense(
            units=self.embedding_dims,
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.l1_trend,
                l2=self.l2_trend))
        self.sea_embedding = tf.keras.layers.Dense(
            units=self.embedding_dims,
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.l1_seasonality,
                l2=self.l2_seasonality))
        self.res_embedding = tf.keras.layers.Dense(
            units=self.embedding_dims,
            kernel_regularizer=tf.keras.regularizers.L1L2(
                l1=self.l1_residual,
                l2=self.l2_residual))

        self.encoder_representation = Representation(
            nr_of_encoder_blocks=self.nr_of_encoder_blocks,
            nr_of_heads=self.nr_of_heads,
            dropout_rate=self.dropout_rate,
            encoder_ffn_units=self.encoder_ffn_units,
            embedding_dims=self.embedding_dims,
            use_time2vec=self.use_time2vec,
            prefer_dense_to_time2vec=prefer_dense_to_time2vec)

        self.unpatcher = tf.keras.layers.Reshape((-1, nr_of_covariates))
        self.timesteps_concatter = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs):

        tre_patch, sea_patch, res_patch, dates = inputs

        # unpatch to get the original shape of the inputs.
        tre_series = self.unpatcher(tre_patch)
        sea_series = self.unpatcher(sea_patch)
        res_series = self.unpatcher(res_patch)

        # embed the patched inputs to get the representation for each component.
        tre_embed = self.tre_embedding(tre_patch)
        sea_embed = self.sea_embedding(sea_patch)
        res_embed = self.res_embedding(res_patch)

        # add prompts if the prompt pool size is greater than 0.
        if self.trend_prompt is not None:
            tre_prompts = self.trend_prompt(tre_series)
            sea_prompts = self.seasonality_prompt(sea_series)
            res_prompts = self.residual_prompt(res_series)

            tre_embed = self.timesteps_concatter([tre_prompts, tre_embed])
            sea_embed = self.timesteps_concatter([sea_prompts, sea_embed])
            res_embed = self.timesteps_concatter([res_prompts, res_embed])

        # get the final representation
        y_cont_temp = self.encoder_representation(
            (tre_embed, sea_embed, res_embed, dates))

        return y_cont_temp

    def get_config(self):
        config = super().get_config()

        config.update({
            'nr_of_covariates': self.nr_of_covariates,
            'nr_of_patches': self.nr_of_patches,
            'nr_of_encoder_blocks': self.nr_of_encoder_blocks,
            'nr_of_heads': self.nr_of_heads,
            'dropout_rate': self.dropout_rate,
            'encoder_ffn_units': self.encoder_ffn_units,
            'embedding_dims': self.embedding_dims,
            'nr_of_timesteps': self.nr_of_timesteps,
            'prompt_pool_size': self.prompt_pool_size,
            'nr_of_most_similar_prompts': self.nr_of_most_similar_prompts,
            'use_time2vec': self.use_time2vec,
            'l1_trend': self.l1_trend,
            'l2_trend': self.l2_trend,
            'l1_seasonality': self.l1_seasonality,
            'l2_seasonality': self.l2_seasonality,
            'l1_residual': self.l1_residual,
            'l2_residual': self.l2_residual,
            'prefer_dense_to_time2vec': self.prefer_dense_to_time2vec,
            'custom_prompt_keys_trend': self.custom_prompt_keys_trend,
            'custom_prompt_keys_seasonality': self.custom_prompt_keys_seasonality,
            'custom_prompt_keys_residual': self.custom_prompt_keys_residual,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class MaskedAutoEncoder(tf.keras.Model):
    '''Masked auto-encoder for pre-training task.'''
    def __init__(
        self,
        representation: RepresentationLearning,
        revIn_tre: ReversibleInstanceNormalization,
        revIn_sea: ReversibleInstanceNormalization,
        revIn_res: ReversibleInstanceNormalization,
        nr_of_timesteps: int,
        nr_of_covariates: int,
        **kwargs):
        super().__init__(**kwargs)

        self.representation = representation
        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res
        self.nr_of_timesteps = nr_of_timesteps
        self.nr_of_covariates = nr_of_covariates

        self.decoder_tre = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            name='decoder_tre')
        self.decoder_sea = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            name='decoder_sea')
        self.decoder_res = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            name='decoder_res')

        self.unpatcher = tf.keras.layers.Reshape((-1, nr_of_covariates))

    def call(self, inputs):

        tre, sea, res, dates = inputs

        z = self.representation((tre, sea, res, dates))

        y_tre = self.decoder_tre(z)
        y_sea = self.decoder_sea(z)
        y_res = self.decoder_res(z)

        y_tre = self.revIn_tre.denormalize((self.unpatcher(tre), y_tre))
        y_sea = self.revIn_sea.denormalize((self.unpatcher(sea), y_sea))
        y_res = self.revIn_res.denormalize((self.unpatcher(res), y_res))

        # compose
        y_composed = y_tre + y_sea + y_res

        return (y_tre, y_sea, y_res, y_composed)


    def get_config(self):
        config = super().get_config()

        config.update({
            'revIn_tre': tf.keras.layers.serialize(self.revIn_tre),
            'revIn_sea': tf.keras.layers.serialize(self.revIn_sea),
            'revIn_res': tf.keras.layers.serialize(self.revIn_res),
            'representation': tf.keras.layers.serialize(self.representation),
            'nr_of_timesteps': self.nr_of_timesteps,
            'nr_of_covariates': self.nr_of_covariates
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['revIn_tre'] = tf.keras.layers.deserialize(config['revIn_tre'])
        config['revIn_sea'] = tf.keras.layers.deserialize(config['revIn_sea'])
        config['revIn_res'] = tf.keras.layers.deserialize(config['revIn_res'])
        config['representation'] = tf.keras.layers.deserialize(config['representation'])

        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class ContrastiveLearning(tf.keras.Model):
    '''Contrastive learning head for pre-training task.'''
    def __init__(
        self,
        representation: RepresentationLearning,
        projection_head_units: int,
        **kwargs):
        super().__init__(**kwargs)

        self.representation = representation
        self.projection_head_units = projection_head_units
        self.projection_head = ProjectionHead(
            projection_head_units,
            name='projection_head')

    def call(self, inputs):
        tre, sea, res, dates = inputs

        z = self.representation((tre, sea, res, dates))

        z_proj = self.projection_head(z)

        return z_proj

    def get_config(self):
        config = super().get_config()

        config.update({
            'representation': tf.keras.layers.serialize(self.representation),
            'projection_head_units': self.projection_head_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['representation'] = tf.keras.layers.deserialize(config['representation'])

        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class PreTraining(tf.keras.Model):
    '''
    Keras model for pre-training purpose.
    This model acts as a foundation model for downstream tasks.
    '''
    def __init__(
        self,
        revIn_tre: ReversibleInstanceNormalization,
        revIn_sea: ReversibleInstanceNormalization,
        revIn_res: ReversibleInstanceNormalization,
        patch_tokenizer: PatchTokenizer,
        representation: RepresentationLearning,
        masked_autoencoder: MaskedAutoEncoder,
        contrastive_learning: ContrastiveLearning,
        nr_of_patches: int,
        mask_rate: float,
        msk_scalar: float,
        nr_of_timesteps: int,
        contrastive_learning_patches: int,
        mae_threshold_comp: float,
        mae_threshold_tre: float,
        mae_threshold_sea: float,
        cl_margin: float,
        force_mae_comp: int,
        force_mae_tre: int,
        force_mae_sea: int,
        force_cl: int,
        **kwargs):

        super(PreTraining, self).__init__(**kwargs)

        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res
        self.patch_tokenizer = patch_tokenizer
        self.representation = representation
        self.masked_autoencoder = masked_autoencoder
        self.contrastive_learning = contrastive_learning
        self.nr_of_patches = nr_of_patches
        self.mask_rate = mask_rate
        self.msk_scalar = msk_scalar
        self.nr_of_timesteps = nr_of_timesteps
        self.contrastive_learning_patches = contrastive_learning_patches
        self.mae_threshold_comp = mae_threshold_comp
        self.mae_threshold_tre = mae_threshold_tre
        self.mae_threshold_sea = mae_threshold_sea
        self.cl_margin = cl_margin
        self.force_mae_comp = force_mae_comp
        self.force_mae_tre = force_mae_tre
        self.force_mae_sea = force_mae_sea
        self.force_cl = force_cl

        self.patch_masker = PatchMasker(msk_scalar=msk_scalar)
        self.timestep_shifter = TimeStepShifter()
        self.timesteps_concatter = tf.keras.layers.Concatenate(axis=1)

        # learning rate tracker
        self.lr_tracker = tf.keras.metrics.Mean(name='lr')

        # losses
        self.loss_tracker_mae_comp = tf.keras.metrics.Mean(name='loss_mae_comp')
        self.loss_tracker_mae_tre = tf.keras.metrics.Mean(name='loss_mae_tre')
        self.loss_tracker_mae_sea = tf.keras.metrics.Mean(name='loss_mae_sea')
        self.loss_tracker_cl = tf.keras.metrics.Mean(name='loss_cl')

        # metrics
        self.mae_tre = tf.keras.metrics.Mean(name='mae_tre')
        self.mae_sea = tf.keras.metrics.Mean(name='mae_sea')
        self.mae_res = tf.keras.metrics.Mean(name='mae_res')
        self.cos_tre = tf.keras.metrics.Mean(name='cos_tre')
        self.cos_sea = tf.keras.metrics.Mean(name='cos_sea')
        self.cos_res = tf.keras.metrics.Mean(name='cos_res')
        self.cos_true = tf.keras.metrics.CosineSimilarity(name='cos_true')
        self.cos_false = tf.keras.metrics.CosineSimilarity(name='cos_false')
        self.mae_composed = tf.keras.metrics.Mean(name='mae_composed')
        self.cosine_similarity = lambda y_true, y_pred: tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1) * -1

    def compile(
        self,
        mae_comp_optimizer: tf.keras.optimizers.Optimizer,
        mae_tre_optimizer: tf.keras.optimizers.Optimizer,
        mae_sea_optimizer: tf.keras.optimizers.Optimizer,
        cl_optimizer: tf.keras.optimizers.Optimizer,
        **kwargs) -> None:

        super().compile(**kwargs)

        self.mae_comp_optimizer = mae_comp_optimizer
        self.mae_tre_optimizer = mae_tre_optimizer
        self.mae_sea_optimizer = mae_sea_optimizer
        self.cl_optimizer = cl_optimizer

    def get_config(self):
        config = super().get_config()

        config.update({
            'revIn_tre': tf.keras.layers.serialize(self.revIn_tre),
            'revIn_sea': tf.keras.layers.serialize(self.revIn_sea),
            'revIn_res': tf.keras.layers.serialize(self.revIn_res),
            'patch_tokenizer': tf.keras.layers.serialize(self.patch_tokenizer),
            'representation': tf.keras.layers.serialize(self.representation),
            'masked_autoencoder': tf.keras.layers.serialize(self.masked_autoencoder),
            'contrastive_learning': tf.keras.layers.serialize(self.contrastive_learning),
            'nr_of_patches': self.nr_of_patches,
            'mask_rate': self.mask_rate,
            'msk_scalar': self.msk_scalar,
            'nr_of_timesteps': self.nr_of_timesteps,
            'contrastive_learning_patches': self.contrastive_learning_patches,
            'mae_threshold_comp': self.mae_threshold_comp,
            'mae_threshold_tre': self.mae_threshold_tre,
            'mae_threshold_sea': self.mae_threshold_sea,
            'cl_margin': self.cl_margin,
            'force_mae_comp': self.force_mae_comp,
            'force_mae_tre': self.force_mae_tre,
            'force_mae_sea': self.force_mae_sea,
            'force_cl': self.force_cl
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['revIn_tre'] = tf.keras.layers.deserialize(config['revIn_tre'])
        config['revIn_sea'] = tf.keras.layers.deserialize(config['revIn_sea'])
        config['revIn_res'] = tf.keras.layers.deserialize(config['revIn_res'])
        config['patch_tokenizer'] = tf.keras.layers.deserialize(config['patch_tokenizer'])
        config['representation'] = tf.keras.layers.deserialize(config['representation'])
        config['masked_autoencoder'] = tf.keras.layers.deserialize(config['masked_autoencoder'])
        config['contrastive_learning'] = tf.keras.layers.deserialize(config['contrastive_learning'])

        return cls(**config)

    def get_compile_config(self):
        return {
            'mae_comp_optimizer': tf.keras.optimizers.serialize(self.mae_comp_optimizer),
            'mae_tre_optimizer': tf.keras.optimizers.serialize(self.mae_tre_optimizer),
            'mae_sea_optimizer': tf.keras.optimizers.serialize(self.mae_sea_optimizer),
            'cl_optimizer': tf.keras.optimizers.serialize(self.cl_optimizer),
        }

    def compile_from_config(self, config):
        self.compile(
            mae_comp_optimizer=tf.keras.optimizers.deserialize(config['mae_comp_optimizer']),
            mae_tre_optimizer=tf.keras.optimizers.deserialize(config['mae_tre_optimizer']),
            mae_sea_optimizer=tf.keras.optimizers.deserialize(config['mae_sea_optimizer']),
            cl_optimizer=tf.keras.optimizers.deserialize(config['cl_optimizer']))

    def generate_mask(self, nr_of_timesteps: int) -> tf.Tensor:
        '''Generates patch indices to mask'''
        nr_of_timesteps_to_mask = tf.cast(
            tf.math.ceil(nr_of_timesteps * self.mask_rate),
            dtype=tf.int32)

        random_tensor = tf.random.uniform(
            shape=(nr_of_timesteps, ),
            minval=0,
            maxval=1)

        sorted_indices = tf.argsort(random_tensor)

        indices_to_mask = sorted_indices[: nr_of_timesteps_to_mask]

        mask = tf.reduce_any(
            tf.equal(tf.range(nr_of_timesteps)[:, tf.newaxis], indices_to_mask),
            axis=1)

        return mask

    def calculate_masked_loss(
        self,
        y_pred: tf.Tensor,
        y_true: tf.Tensor,
        mask: tf.Tensor,
        loss_fn: Callable) -> tf.Tensor:
        '''
        Calculates loss only for masked patches.
        y_true and y_pred are patched.
        based on masked patches, the loss is calculated.

        args:
            y_pred (None, timesteps, covariates) - predicted output
                which is unpatched.
            y_true (None, timesteps, covariates)- actual output
                which is unpatched.
            mask (timesteps,) - boolean mask for patches. True values
                indicate the patches to calculate loss.
            loss_fn (tf.keras.losses) - loss function

        returns
            loss (int) - calculated loss
        '''

        true_patched = self.patch_tokenizer(y_true)
        pred_patched = self.patch_tokenizer(y_pred)

        true_masked = tf.boolean_mask(
            tensor=true_patched,
            mask=mask,
            axis=1)
        pred_masked = tf.boolean_mask(
            tensor=pred_patched,
            mask=mask,
            axis=1)

        loss = loss_fn(y_pred=pred_masked, y_true=true_masked)

        return tf.reduce_mean(loss)

    def _compute_reconstruction_losses(
        self,
        y_pred_tre: tf.Tensor, y_pred_sea: tf.Tensor, y_pred_res: tf.Tensor, y_pred_composed: tf.Tensor,
        anchor_tre: tf.Tensor, anchor_sea: tf.Tensor, anchor_res: tf.Tensor, anchor_composed: tf.Tensor,
        mask: tf.Tensor, loss_fn: Callable) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        loss_comp = self.calculate_masked_loss(
            y_pred=y_pred_composed, y_true=anchor_composed, mask=mask, loss_fn=loss_fn)
        loss_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre, y_true=anchor_tre, mask=mask, loss_fn=loss_fn)
        loss_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea, y_true=anchor_sea, mask=mask, loss_fn=loss_fn)
        return loss_comp, loss_tre, loss_sea

    def _update_diagnostic_metrics(
        self,
        y_pred_tre: tf.Tensor, y_pred_sea: tf.Tensor, y_pred_res: tf.Tensor, y_pred_composed: tf.Tensor,
        anchor_tre: tf.Tensor, anchor_sea: tf.Tensor, anchor_res: tf.Tensor, anchor_composed: tf.Tensor,
        mask: tf.Tensor) -> None:
        mae_composed = self.calculate_masked_loss(
            y_pred=y_pred_composed, y_true=anchor_composed, mask=mask,
            loss_fn=tf.keras.losses.mean_absolute_error)
        mae_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre, y_true=anchor_tre, mask=mask,
            loss_fn=tf.keras.losses.mean_absolute_error)
        mae_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea, y_true=anchor_sea, mask=mask,
            loss_fn=tf.keras.losses.mean_absolute_error)
        mae_res = self.calculate_masked_loss(
            y_pred=y_pred_res, y_true=anchor_res, mask=mask,
            loss_fn=tf.keras.losses.mean_absolute_error)
        cos_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre, y_true=anchor_tre, mask=mask,
            loss_fn=self.cosine_similarity)
        cos_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea, y_true=anchor_sea, mask=mask,
            loss_fn=self.cosine_similarity)
        cos_res = self.calculate_masked_loss(
            y_pred=y_pred_res, y_true=anchor_res, mask=mask,
            loss_fn=self.cosine_similarity)
        self.mae_composed.update_state(mae_composed)
        self.mae_tre.update_state(mae_tre)
        self.mae_sea.update_state(mae_sea)
        self.mae_res.update_state(mae_res)
        self.cos_tre.update_state(cos_tre)
        self.cos_sea.update_state(cos_sea)
        self.cos_res.update_state(cos_res)

    def _update_reconstruction_loss_trackers(self, loss_mae_comp: tf.Tensor, loss_mae_tre: tf.Tensor, loss_mae_sea: tf.Tensor) -> None:
        self.loss_tracker_mae_comp.update_state(loss_mae_comp)
        self.loss_tracker_mae_tre.update_state(loss_mae_tre)
        self.loss_tracker_mae_sea.update_state(loss_mae_sea)

    def _compute_contrastive_losses(self, y_logits_anchor: tf.Tensor, y_logits_true: tf.Tensor, y_logits_false: tf.Tensor) -> tf.Tensor:
        distance_true = tf.reduce_sum(tf.square(y_logits_anchor - y_logits_true), -1)
        distance_false = tf.reduce_sum(tf.square(y_logits_anchor - y_logits_false), -1)
        return tf.maximum(distance_true - distance_false + self.cl_margin, 0.0)

    def _update_contrastive_loss_trackers(self, loss_cl: tf.Tensor) -> None:
        self.loss_tracker_cl.update_state(loss_cl)

    def _update_contrastive_metrics(self, y_logits_anchor: tf.Tensor, y_logits_true: tf.Tensor, y_logits_false: tf.Tensor) -> None:
        self.cos_true.update_state(y_true=y_logits_anchor, y_pred=y_logits_true)
        self.cos_false.update_state(y_true=y_logits_anchor, y_pred=y_logits_false)

    def augment_pairs(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        Augments an input. In each augmentation, different patches are
            masked & shifted randomly.
        Only lookbacks are masked. Forecast are not masked.
        In each augmentation, forecast patches are shifted (rolled) by
            random amount.

        returns: tuples of 6 elements. Each element contains merged
            lookback and forecast patches.
        '''
        x_tre, x_sea, x_res = data
        x_lb_tre = x_tre[:, :self.contrastive_learning_patches]
        x_lb_sea = x_sea[:, :self.contrastive_learning_patches]
        x_lb_res = x_res[:, :self.contrastive_learning_patches]
        x_fc_tre = x_tre[:, self.contrastive_learning_patches:]
        x_fc_sea = x_sea[:, self.contrastive_learning_patches:]
        x_fc_res = x_res[:, self.contrastive_learning_patches:]

        nr_of_forecast_patches = tf.shape(x_fc_tre)[1]

        # mask
        x_fc_tre_msk, x_fc_sea_msk, x_fc_res_msk = self.patch_masker(
            (x_fc_tre, x_fc_sea, x_fc_res, mask))

        x_tre_true = self.timesteps_concatter(
            [x_lb_tre, x_fc_tre_msk])
        x_sea_true = self.timesteps_concatter(
            [x_lb_sea, x_fc_sea_msk])
        x_res_true = self.timesteps_concatter(
            [x_lb_res, x_fc_res_msk])

        # shift
        i = tf.random.uniform(
            shape=[],
            minval=1,
            maxval=nr_of_forecast_patches,
            dtype=tf.int32)
        x_fc_tre_sft, x_fc_sea_sft, x_fc_res_sft = self.timestep_shifter(
            (x_fc_tre, x_fc_sea, x_fc_res, i))
        x_tre_false = self.timesteps_concatter(
            [x_lb_tre, x_fc_tre_sft])
        x_sea_false = self.timesteps_concatter(
            [x_lb_sea, x_fc_sea_sft])
        x_res_false = self.timesteps_concatter(
            [x_lb_res, x_fc_res_sft])

        return (x_tre_true,
                x_sea_true,
                x_res_true,
                x_tre_false,
                x_sea_false,
                x_res_false)

    def get_tasks_to_train(
        self,
        mae_comp: float,
        mae_tre: float,
        mae_sea: float) -> tf.Tensor:
        '''
        identifies the training task.
        firstly, it identifies the proposed sequence.
        later, enforcement rules are introduced.
        args:
            mae_comp - mean absolute error of composed series.
            mae_tre - mean absolute error of trend component.
            mae_sea - mean absolute error of seasonality component.

        returns:
            tasks - list of tasks to train
        '''

        tasks = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True)

        # proposed sequence of training tasks
        #############################################
        achieve_comp = tf.less_equal(mae_comp, self.mae_threshold_comp)
        achieve_tre = tf.less_equal(mae_tre, self.mae_threshold_tre)
        achieve_sea = tf.less_equal(mae_sea, self.mae_threshold_sea)

        msk_autoenc_comp = tf.logical_not(achieve_comp)

        msk_autoenc_tre = tf.logical_and(
            tf.logical_not(achieve_tre),
            achieve_comp)

        msk_autoenc_sea = tf.reduce_all(
            [
                tf.logical_not(achieve_sea),
                achieve_comp,
                achieve_tre
            ])

        cl = tf.logical_not(tf.equal(self.force_cl, -1))
        #############################################

        # introduce force to train
        #############################################
        msk_autoenc_comp = tf.logical_or(
            msk_autoenc_comp,
            tf.equal(self.force_mae_comp, 1))

        msk_autoenc_tre = tf.logical_or(
            msk_autoenc_tre,
            tf.equal(self.force_mae_tre, 1))

        msk_autoenc_sea = tf.logical_or(
            msk_autoenc_sea,
            tf.equal(self.force_mae_sea, 1))
        #############################################

        # introduce force to not train
        #############################################
        msk_autoenc_comp = tf.logical_and(
            msk_autoenc_comp,
            tf.logical_not(tf.equal(self.force_mae_comp, -1)))

        msk_autoenc_tre = tf.logical_and(
            msk_autoenc_tre,
            tf.logical_not(tf.equal(self.force_mae_tre, -1)))

        msk_autoenc_sea = tf.logical_and(
            msk_autoenc_sea,
            tf.logical_not(tf.equal(self.force_mae_sea, -1)))
        #############################################

        tasks = tf.cond(
            msk_autoenc_comp,
            lambda: tasks.write(tasks.size(), tf.constant('msk_autoenc_comp')),
            lambda: tasks)

        tasks = tf.cond(
            msk_autoenc_tre,
            lambda: tasks.write(tasks.size(), tf.constant('msk_autoenc_tre')),
            lambda: tasks)

        tasks = tf.cond(
            msk_autoenc_sea,
            lambda: tasks.write(tasks.size(), tf.constant('msk_autoenc_sea')),
            lambda: tasks)

        tasks = tf.cond(
            cl,
            lambda: tasks.write(tasks.size(), tf.constant('cl')),
            lambda: tasks)

        # if there is no training task assigned,
        # then train with masked autoencoder composed.
        assigned_to_at_least_one_task = tf.reduce_any(
            [
                msk_autoenc_comp,
                msk_autoenc_tre,
                msk_autoenc_sea,
                cl
            ])

        tasks = tf.cond(
            tf.logical_not(assigned_to_at_least_one_task),
            lambda: tasks.write(tasks.size(), tf.constant('msk_autoenc_comp')),
            lambda: tasks)

        return tasks.stack()

    @tf.function()
    def train_step(self, data):
        '''
        args:
            anchor_tre: (None, timesteps, covariates)
            anchor_sea: (None, timesteps, covariates)
            anchor_res: (None, timesteps, covariates)
            dates: (None, features)

        trains a step in two phases:
            1. masked patch prediction
            2. contrastive learning
        '''

        anchor_tre, anchor_sea, anchor_res, _ = data
        anchor_composed = anchor_tre + anchor_sea + anchor_res

        # generate masks for masked autoencoder task and contrastive learning task
        mask = self.generate_mask(nr_of_timesteps=self.nr_of_patches)
        cl_mask = self.generate_mask(nr_of_timesteps=(self.nr_of_patches - self.contrastive_learning_patches))

        # masked auto-encoder (mae)
        y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
            self(data, training=True, mask=mask, task="reconstruction")

        mae_comp, mae_tre, mae_sea = self._compute_reconstruction_losses(
            y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
            anchor_tre, anchor_sea, anchor_res, anchor_composed,
            mask, loss_fn=tf.keras.losses.mean_absolute_error)

        tasks_to_train = self.get_tasks_to_train(mae_comp, mae_tre, mae_sea)

        if tf.reduce_any(
            tf.equal(
                tasks_to_train,
                tf.constant('msk_autoenc_comp'))):
            with tf.GradientTape() as tape:
                y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = self(data, training=True, mask=mask, task="reconstruction")

                # compute the loss values
                loss_mae_comp, loss_mae_tre, loss_mae_sea = self._compute_reconstruction_losses(
                    y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
                    anchor_tre, anchor_sea, anchor_res, anchor_composed,
                    mask, loss_fn=tf.keras.losses.mean_squared_error)

            # compute gradients
            mae_trainable_vars = self.revIn_tre.trainable_variables + \
                self.revIn_sea.trainable_variables + \
                self.revIn_res.trainable_variables + \
                self.masked_autoencoder.trainable_variables

            mae_gradients = tape.gradient(loss_mae_comp, mae_trainable_vars)

            # update weights
            self.mae_comp_optimizer.apply_gradients(zip(mae_gradients, mae_trainable_vars))

            # log losses
            self._update_reconstruction_loss_trackers(loss_mae_comp, loss_mae_tre, loss_mae_sea)

        if tf.reduce_any(
            tf.equal(
                tasks_to_train,
                tf.constant('msk_autoenc_tre'))):
            with tf.GradientTape() as tape:
                y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = self(data, training=True, mask=mask, task="reconstruction")

                # compute the loss values
                loss_mae_comp, loss_mae_tre, loss_mae_sea = self._compute_reconstruction_losses(
                    y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
                    anchor_tre, anchor_sea, anchor_res, anchor_composed,
                    mask, loss_fn=tf.keras.losses.mean_squared_error)

            # compute gradients
            mae_trainable_vars = self.revIn_tre.trainable_variables + \
                self.masked_autoencoder.trainable_variables
            mae_gradients = tape.gradient(loss_mae_tre, mae_trainable_vars)

            # update weights (skip vars with no gradient path to loss_mae_tre)
            self.mae_tre_optimizer.apply_gradients(
                (g, v) for g, v in zip(mae_gradients, mae_trainable_vars) if g is not None)

            # log losses
            self._update_reconstruction_loss_trackers(loss_mae_comp, loss_mae_tre, loss_mae_sea)

        if tf.reduce_any(
            tf.equal(
                tasks_to_train,
                tf.constant('msk_autoenc_sea'))):
            with tf.GradientTape() as tape:
                y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = self(data, training=True, mask=mask, task="reconstruction")

                # compute the loss values
                loss_mae_comp, loss_mae_tre, loss_mae_sea = self._compute_reconstruction_losses(
                    y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
                    anchor_tre, anchor_sea, anchor_res, anchor_composed,
                    mask, loss_fn=tf.keras.losses.mean_squared_error)

            # compute gradients
            mae_trainable_vars = self.revIn_sea.trainable_variables + \
                self.masked_autoencoder.trainable_variables
            mae_gradients = tape.gradient(loss_mae_sea, mae_trainable_vars)

            # update weights (skip vars with no gradient path to loss_mae_sea)
            self.mae_sea_optimizer.apply_gradients(
                (g, v) for g, v in zip(mae_gradients, mae_trainable_vars) if g is not None)

            # log losses
            self._update_reconstruction_loss_trackers(loss_mae_comp, loss_mae_tre, loss_mae_sea)

        if tf.reduce_any(
            tf.equal(
                tasks_to_train,
                tf.constant('cl'))):

            # contrastive learning
            with tf.GradientTape() as tape:
                y_logits_false, y_logits_true, y_logits_anchor = self(data, training=True, mask=cl_mask, task="contrastive")

                loss_cl = self._compute_contrastive_losses(y_logits_anchor, y_logits_true, y_logits_false)

            # compute gradients for contrastive learning
            cl_trainable_vars = \
                self.revIn_tre.trainable_variables + \
                self.revIn_sea.trainable_variables + \
                self.revIn_res.trainable_variables + \
                self.contrastive_learning.trainable_variables

            cl_gradients = tape.gradient(loss_cl, cl_trainable_vars)

            # update weights
            self.cl_optimizer.apply_gradients(zip(cl_gradients, cl_trainable_vars))

            self._update_contrastive_loss_trackers(loss_cl)
            self._update_contrastive_metrics(y_logits_anchor, y_logits_true, y_logits_false)

        self._update_diagnostic_metrics(
            y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
            anchor_tre, anchor_sea, anchor_res, anchor_composed, mask)

        self.lr_tracker.update_state(self.cl_optimizer.lr)

        dic = {
            'loss_mae_comp': self.loss_tracker_mae_comp.result(),
            'loss_mae_tre': self.loss_tracker_mae_tre.result(),
            'loss_mae_sea': self.loss_tracker_mae_sea.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'mae_res': self.mae_res.result(),
            'mae_composed': self.mae_composed.result(),
            'cos_tre': self.cos_tre.result(),
            'cos_sea': self.cos_sea.result(),
            'cos_res': self.cos_res.result(),
            'cos_true': self.cos_true.result(),
            'cos_false': self.cos_false.result(),
            'lr': self.lr_tracker.result()}

        return dic

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker_mae_comp,
            self.loss_tracker_mae_tre,
            self.loss_tracker_mae_sea,
            self.loss_tracker_cl,
            self.lr_tracker,
            self.mae_composed,
            self.mae_tre,
            self.mae_sea,
            self.mae_res,
            self.cos_tre,
            self.cos_sea,
            self.cos_res,
            self.cos_true,
            self.cos_false]

    def test_step(self, data):
        anchor_tre, anchor_sea, anchor_res, dates = data
        anchor_composed = anchor_tre + anchor_sea + anchor_res

        # get mask
        mask = self.generate_mask(nr_of_timesteps=self.nr_of_patches)
        cl_mask = self.generate_mask(nr_of_timesteps=(self.nr_of_patches - self.contrastive_learning_patches))

        y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
            self(data, training=False, mask=mask, task="reconstruction")

        loss_mae_comp, loss_mae_tre, loss_mae_sea = self._compute_reconstruction_losses(
            y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
            anchor_tre, anchor_sea, anchor_res, anchor_composed,
            mask, loss_fn=tf.keras.losses.mean_squared_error)

        self._update_reconstruction_loss_trackers(loss_mae_comp, loss_mae_tre, loss_mae_sea)

        self._update_diagnostic_metrics(
            y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
            anchor_tre, anchor_sea, anchor_res, anchor_composed, mask)

        # calculate contrastive learning logits
        y_logits_false, y_logits_true, y_logits_anchor = \
            self(data, training=False, mask=cl_mask, task="contrastive")

        loss_cl = self._compute_contrastive_losses(y_logits_anchor, y_logits_true, y_logits_false)

        self._update_contrastive_loss_trackers(loss_cl)
        self._update_contrastive_metrics(y_logits_anchor, y_logits_true, y_logits_false)

        dic = {
            'loss_mae_tre': self.loss_tracker_mae_tre.result(),
            'loss_mae_sea': self.loss_tracker_mae_sea.result(),
            'loss_mae_comp': self.loss_tracker_mae_comp.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'mae_res': self.mae_res.result(),
            'mae_composed': self.mae_composed.result(),
            'cos_tre': self.cos_tre.result(),
            'cos_sea': self.cos_sea.result(),
            'cos_res': self.cos_res.result(),
            'cos_true': self.cos_true.result(),
            'cos_false': self.cos_false.result()}

        return dic

    def predict_step(self, data):
        mask = self.generate_mask(nr_of_timesteps=self.nr_of_patches)
        y_pred_tre, y_pred_sea, y_pred_res, _ = \
            self(data, training=False, mask=mask, task="reconstruction")

        return y_pred_tre, y_pred_sea, y_pred_res, mask

    def call(self, inputs, training=False, mask=None, task="reconstruction"):
        '''
        args:
            tre: (None, timesteps, covariates)
            sea: (None, timesteps, covariates)
            res: (None, timesteps, covariates)
            dates: (None, features)
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

        if task == "contrastive":
            tre_true, sea_true, res_true, tre_false, sea_false, res_false = self.augment_pairs((tre_patch, sea_patch, res_patch), mask)

            y_logits_true = self.contrastive_learning((tre_true, sea_true, res_true, dates))
            y_logits_false = self.contrastive_learning((tre_false, sea_false, res_false, dates))
            y_logits_anchor = self.contrastive_learning((tre_patch, sea_patch, res_patch, dates))

            return y_logits_false, y_logits_true, y_logits_anchor

        if task == "reconstruction":
            tre_patch, sea_patch, res_patch = self.patch_masker((tre_patch, sea_patch, res_patch, mask))

            return self.masked_autoencoder((tre_patch, sea_patch, res_patch, dates))


class PreTrainingWeightedLoss(PreTraining):
    '''
    PreTraining variant that combines all losses into a single weighted sum
    and performs one backward pass per step instead of one per task.
    '''

    def __init__(
        self,
        revIn_tre: ReversibleInstanceNormalization,
        revIn_sea: ReversibleInstanceNormalization,
        revIn_res: ReversibleInstanceNormalization,
        patch_tokenizer: PatchTokenizer,
        representation: RepresentationLearning,
        masked_autoencoder: MaskedAutoEncoder,
        contrastive_learning: ContrastiveLearning,
        nr_of_patches: int,
        mask_rate: float,
        msk_scalar: float,
        nr_of_timesteps: int,
        contrastive_learning_patches: int,
        mae_threshold_comp: float,
        mae_threshold_tre: float,
        mae_threshold_sea: float,
        cl_margin: float,
        force_mae_comp: int,
        force_mae_tre: int,
        force_mae_sea: int,
        force_cl: int,
        w_comp: float = 1.0,
        w_tre: float = 1.0,
        w_sea: float = 1.0,
        w_cl: float = 1.0,
        **kwargs):

        super().__init__(
            revIn_tre=revIn_tre,
            revIn_sea=revIn_sea,
            revIn_res=revIn_res,
            patch_tokenizer=patch_tokenizer,
            representation=representation,
            masked_autoencoder=masked_autoencoder,
            contrastive_learning=contrastive_learning,
            nr_of_patches=nr_of_patches,
            mask_rate=mask_rate,
            msk_scalar=msk_scalar,
            nr_of_timesteps=nr_of_timesteps,
            contrastive_learning_patches=contrastive_learning_patches,
            mae_threshold_comp=mae_threshold_comp,
            mae_threshold_tre=mae_threshold_tre,
            mae_threshold_sea=mae_threshold_sea,
            cl_margin=cl_margin,
            force_mae_comp=force_mae_comp,
            force_mae_tre=force_mae_tre,
            force_mae_sea=force_mae_sea,
            force_cl=force_cl,
            **kwargs)

        self.w_comp = w_comp
        self.w_tre = w_tre
        self.w_sea = w_sea
        self.w_cl = w_cl

        self.loss_tracker_weighted = tf.keras.metrics.Mean(name='loss_weighted')

    def compile(self, optimizer: tf.keras.optimizers.Optimizer, **kwargs) -> None:
        # Skip PreTraining.compile (which expects 4 optimizers) and go straight
        # to tf.keras.Model.compile so self.optimizer is set normally.
        super(PreTraining, self).compile(optimizer=optimizer, **kwargs)

    def get_compile_config(self) -> dict:
        return {'optimizer': tf.keras.optimizers.serialize(self.optimizer)}

    def compile_from_config(self, config: dict) -> None:
        self.compile(optimizer=tf.keras.optimizers.deserialize(config['optimizer']))

    @property
    def metrics(self):
        return [
            self.loss_tracker_weighted,
            self.loss_tracker_mae_comp,
            self.loss_tracker_mae_tre,
            self.loss_tracker_mae_sea,
            self.loss_tracker_cl,
            self.lr_tracker,
            self.mae_composed,
            self.mae_tre,
            self.mae_sea,
            self.mae_res,
            self.cos_tre,
            self.cos_sea,
            self.cos_res,
            self.cos_true,
            self.cos_false]

    def train_step(self, data):
        anchor_tre, anchor_sea, anchor_res, _ = data
        anchor_composed = anchor_tre + anchor_sea + anchor_res

        mask = self.generate_mask(nr_of_timesteps=self.nr_of_patches)
        cl_mask = self.generate_mask(nr_of_timesteps=(self.nr_of_patches - self.contrastive_learning_patches))

        with tf.GradientTape() as tape:
            y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
                self(data, training=True, mask=mask, task="reconstruction")
            y_logits_false, y_logits_true, y_logits_anchor = \
                self(data, training=True, mask=cl_mask, task="contrastive")

            loss_mae_comp, loss_mae_tre, loss_mae_sea = self._compute_reconstruction_losses(
                y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
                anchor_tre, anchor_sea, anchor_res, anchor_composed,
                mask, loss_fn=tf.keras.losses.mean_squared_error)

            loss_cl = self._compute_contrastive_losses(y_logits_anchor, y_logits_true, y_logits_false)

            loss = (self.w_comp * loss_mae_comp +
                    self.w_tre  * loss_mae_tre  +
                    self.w_sea  * loss_mae_sea  +
                    self.w_cl   * loss_cl)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update loss trackers
        self.loss_tracker_weighted.update_state(loss)
        self._update_reconstruction_loss_trackers(loss_mae_comp, loss_mae_tre, loss_mae_sea)
        self._update_contrastive_loss_trackers(loss_cl)
        self._update_contrastive_metrics(y_logits_anchor, y_logits_true, y_logits_false)

        self._update_diagnostic_metrics(
            y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed,
            anchor_tre, anchor_sea, anchor_res, anchor_composed, mask)

        self.lr_tracker.update_state(self.optimizer.lr)

        return {
            'loss_weighted': self.loss_tracker_weighted.result(),
            'loss_mae_comp': self.loss_tracker_mae_comp.result(),
            'loss_mae_tre':  self.loss_tracker_mae_tre.result(),
            'loss_mae_sea':  self.loss_tracker_mae_sea.result(),
            'loss_cl':       self.loss_tracker_cl.result(),
            'mae_tre':       self.mae_tre.result(),
            'mae_sea':       self.mae_sea.result(),
            'mae_res':       self.mae_res.result(),
            'mae_composed':  self.mae_composed.result(),
            'cos_tre':       self.cos_tre.result(),
            'cos_sea':       self.cos_sea.result(),
            'cos_res':       self.cos_res.result(),
            'cos_true':      self.cos_true.result(),
            'cos_false':     self.cos_false.result(),
            'lr':            self.lr_tracker.result()}

    def get_config(self):
        config = super().get_config()
        config.update({
            'w_comp': self.w_comp,
            'w_tre':  self.w_tre,
            'w_sea':  self.w_sea,
            'w_cl':   self.w_cl})
        return config


def build_model(
    nr_of_covariates: int,
    patch_size: int,
    nr_of_encoder_blocks: int,
    nr_of_heads: int,
    dropout_rate: float,
    encoder_ffn_units: int,
    embedding_dims: int,
    projection_head_units: int,
    mask_rate: float,
    msk_scalar: float,
    nr_of_timesteps: int,
    contrastive_learning_patches: int,
    mae_threshold_comp: float,
    mae_threshold_tre: float,
    mae_threshold_sea: float,
    cl_margin: float,
    prompt_pool_size: int,
    nr_of_most_similar_prompts: int,
    use_time2vec: bool,
    force_mae_comp: int,
    force_mae_tre: int,
    force_mae_sea: int,
    force_cl: int,
    l1_trend: float,
    l2_trend: float,
    l1_seasonality: float,
    l2_seasonality: float,
    l1_residual: float,
    l2_residual: float,
    prefer_dense_to_time2vec: bool,
    custom_prompt_keys_trend: list,
    custom_prompt_keys_seasonality: list,
    custom_prompt_keys_residual: list,
    w_comp: Optional[float] = None,
    w_tre: Optional[float] = None,
    w_sea: Optional[float] = None,
    w_cl: Optional[float] = None) -> Union[PreTraining, PreTrainingWeightedLoss]:

    nr_of_patches = nr_of_timesteps // patch_size

    revIn_tre = ReversibleInstanceNormalization(
        nr_of_covariates=nr_of_covariates,
        epsilon=1e-6)

    revIn_sea = ReversibleInstanceNormalization(
        nr_of_covariates=nr_of_covariates,
        epsilon=1e-6)

    revIn_res = ReversibleInstanceNormalization(
        nr_of_covariates=nr_of_covariates,
        epsilon=1e-6)

    patch_tokenizer = PatchTokenizer(
        patch_size=patch_size,
        nr_of_covariates=nr_of_covariates)

    representation = RepresentationLearning(
        nr_of_covariates=nr_of_covariates,
        nr_of_patches=nr_of_patches,
        nr_of_encoder_blocks=nr_of_encoder_blocks,
        nr_of_heads=nr_of_heads,
        dropout_rate=dropout_rate,
        encoder_ffn_units=encoder_ffn_units,
        embedding_dims=embedding_dims,
        nr_of_timesteps=nr_of_timesteps,
        prompt_pool_size=prompt_pool_size,
        nr_of_most_similar_prompts=nr_of_most_similar_prompts,
        use_time2vec=use_time2vec,
        l1_trend=l1_trend,
        l2_trend=l2_trend,
        l1_seasonality=l1_seasonality,
        l2_seasonality=l2_seasonality,
        l1_residual=l1_residual,
        l2_residual=l2_residual,
        prefer_dense_to_time2vec=prefer_dense_to_time2vec,
        custom_prompt_keys_trend=custom_prompt_keys_trend,
        custom_prompt_keys_seasonality=custom_prompt_keys_seasonality,
        custom_prompt_keys_residual=custom_prompt_keys_residual)

    masked_autoencoder = MaskedAutoEncoder(
        representation=representation,
        revIn_tre=revIn_tre,
        revIn_sea=revIn_sea,
        revIn_res=revIn_res,
        nr_of_timesteps=nr_of_timesteps,
        nr_of_covariates=nr_of_covariates)

    contrastive_learning = ContrastiveLearning(
        representation=representation,
        projection_head_units=projection_head_units)

    shared_kwargs = dict(
        revIn_tre=revIn_tre,
        revIn_sea=revIn_sea,
        revIn_res=revIn_res,
        patch_tokenizer=patch_tokenizer,
        representation=representation,
        masked_autoencoder=masked_autoencoder,
        contrastive_learning=contrastive_learning,
        nr_of_patches=nr_of_patches,
        mask_rate=mask_rate,
        msk_scalar=msk_scalar,
        nr_of_timesteps=nr_of_timesteps,
        contrastive_learning_patches=contrastive_learning_patches,
        mae_threshold_comp=mae_threshold_comp,
        mae_threshold_tre=mae_threshold_tre,
        mae_threshold_sea=mae_threshold_sea,
        cl_margin=cl_margin,
        force_mae_comp=force_mae_comp,
        force_mae_tre=force_mae_tre,
        force_mae_sea=force_mae_sea,
        force_cl=force_cl)

    use_weighted = all(w is not None for w in [w_comp, w_tre, w_sea, w_cl])
    if use_weighted:
        model = PreTrainingWeightedLoss(
            **shared_kwargs,
            w_comp=w_comp,
            w_tre=w_tre,
            w_sea=w_sea,
            w_cl=w_cl)
    else:
        model = PreTraining(**shared_kwargs)

    return model


def compile_model(model: Union[PreTraining, PreTrainingWeightedLoss], clip_norm: float) -> Union[PreTraining, PreTrainingWeightedLoss]:
    if isinstance(model, PreTrainingWeightedLoss):
        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=clip_norm))
    else:
        model.compile(
            mae_comp_optimizer=tf.keras.optimizers.Adam(clipnorm=clip_norm),
            mae_tre_optimizer=tf.keras.optimizers.Adam(clipnorm=clip_norm),
            mae_sea_optimizer=tf.keras.optimizers.Adam(clipnorm=clip_norm),
            cl_optimizer=tf.keras.optimizers.Adam(clipnorm=clip_norm))

    return model


def get_callbacks(
    embedding_dims: int,
    warmup_steps: int,
    scale_factor: float,
    patience: int,
    warmup_epochs_early_stopping: int) -> list:

    learning_rate_callback = LearningRateCallback(
        d_model=embedding_dims,
        warmup_steps=warmup_steps,
        scale_factor=scale_factor)

    ram_cleaner_callback = RamCleaner()

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae_composed',
        patience=patience,
        start_from_epoch=warmup_epochs_early_stopping,
        restore_best_weights=True)

    timing_callback = TimingCallback()

    callbacks = [
        terminate_on_nan_callback,
        ram_cleaner_callback,
        learning_rate_callback,
        early_stopping,
        timing_callback]

    return callbacks