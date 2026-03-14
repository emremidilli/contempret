import gc

import keras_tuner as kt

import tensorflow as tf

from tsf_model.models import PreTraining

from utils import (
    LearningRateCallback,
    MaskingCallback)

def build_model(
        hp,
        nr_of_timesteps,
        nr_of_covariates,
        use_time2vec):
    '''
    function that builds model for tuning.
    during pre-training, all self-supervised tasks are forced to be trained.
    '''
    # clear previous models
    tf.keras.backend.clear_session()

    # define constants
    MASK_RATE = 0.40
    MASK_SCALAR = 0.0
    MAE_THRESHOLD_COMP = 1.0
    MAE_THRESHOLD_TRE = 1.0
    MAE_THRESHOLD_SEA = 1.0
    CL_MARGIN = 0.25
    FORCE_MAE_COMP = 1
    FORCE_MAE_TRE = 1
    FORCE_MAE_SEA = 1
    FORCE_CL = 1
    PREFER_DENSE_TO_TIME2VEC = False
    CUSTOM_PROMPT_KEYS_TREND = None
    CUSTOM_PROMPT_KEYS_SEASONALITY = None
    CUSTOM_PROMPT_KEYS_RESIDUAL = None

    # define hp parameters
    nr_of_encoder_blocks = hp.Int('nr_of_encoder_blocks', 2, 8, step=1)
    dropout_rate = \
        hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    encoder_ffn_units = \
        hp.Int('encoder_ffn_units', min_value=64, max_value=512, step=64)
    embedding_dims = \
        hp.Choice('embedding_dims', values=[64, 128, 256, 512])
    nr_of_heads = \
        hp.Choice('nr_of_heads', values=[2, 4, 8, 16])
    projection_head = \
        hp.Choice('projection_head', values=[8, 16, 32, 64])
    prompt_pool_size = \
        hp.Choice('prompt_pool_size', values=[10, 20, 30, 40, 50])

    patch_size = hp.Choice('patch_size', values=[4, 8, 12, 24])
    nr_of_most_similar_prompts = \
        hp.Int('nr_of_most_similar_prompts', min_value=1, max_value=12, step=1)

    l1_trend = hp.Float(
        'l1_trend',
        min_value=1e-6,
        max_value=1e-2,
        sampling='log')
    l2_trend = hp.Float(
        'l2_trend',
        min_value=1e-6,
        max_value=1e-2,
        sampling='log')
    l1_seasonality = hp.Float(
        'l1_seasonality',
        min_value=1e-6,
        max_value=1e-2,
        sampling='log')
    l2_seasonality = hp.Float(
        'l2_seasonality',
        min_value=1e-6,
        max_value=1e-2,
        sampling='log')
    l1_residual = hp.Float(
        'l1_residual',
        min_value=1e-6,
        max_value=1e-2,
        sampling='log')
    l2_residual = hp.Float(
        'l2_residual',
        min_value=1e-6,
        max_value=1e-2,
        sampling='log')

    # initialize optimizers
    mae_comp_optimizer = tf.keras.optimizers.Adam()
    mae_tre_optimizer = tf.keras.optimizers.Adam()
    mae_sea_optimizer = tf.keras.optimizers.Adam()

    cl_optimizer = tf.keras.optimizers.Adam()

    # define model
    contrastive_learning_patches = int(nr_of_timesteps * (MASK_RATE))
    contrastive_learning_patches = \
        int(contrastive_learning_patches / patch_size)

    masked_auto_encoder_patches = int(nr_of_timesteps / patch_size)

    model = PreTraining(
        nr_of_covariates=nr_of_covariates,
        patch_size=patch_size,
        nr_of_encoder_blocks=nr_of_encoder_blocks,
        nr_of_heads=nr_of_heads,
        dropout_rate=dropout_rate,
        encoder_ffn_units=encoder_ffn_units,
        embedding_dims=embedding_dims,
        projection_head_units=projection_head,
        msk_scalar=MASK_SCALAR,
        nr_of_timesteps=nr_of_timesteps,
        contrastive_learning_patches=contrastive_learning_patches,
        mae_threshold_comp=MAE_THRESHOLD_COMP,
        mae_threshold_tre=MAE_THRESHOLD_TRE,
        mae_threshold_sea=MAE_THRESHOLD_SEA,
        cl_margin=CL_MARGIN,
        prompt_pool_size=prompt_pool_size,
        nr_of_most_similar_prompts=nr_of_most_similar_prompts,
        use_time2vec=use_time2vec,
        force_mae_comp=FORCE_MAE_COMP,
        force_mae_tre=FORCE_MAE_TRE,
        force_mae_sea=FORCE_MAE_SEA,
        force_cl=FORCE_CL,
        l1_trend=l1_trend,
        l2_trend=l2_trend,
        l1_seasonality=l1_seasonality,
        l2_seasonality=l2_seasonality,
        l1_residual=l1_residual,
        l2_residual=l2_residual,
        prefer_dense_to_time2vec=PREFER_DENSE_TO_TIME2VEC,
        custom_prompt_keys_trend=CUSTOM_PROMPT_KEYS_TREND,
        custom_prompt_keys_seasonality=CUSTOM_PROMPT_KEYS_SEASONALITY,
        custom_prompt_keys_residual=CUSTOM_PROMPT_KEYS_RESIDUAL)

    model.compile(
        mae_comp_optimizer=mae_comp_optimizer,
        mae_tre_optimizer=mae_tre_optimizer,
        mae_sea_optimizer=mae_sea_optimizer,
        cl_optimizer=cl_optimizer)

    return model


def get_callbacks(hp, nr_of_timesteps):
    '''
    returns the callbacks for hyperparameter tuning.
    '''
    MASK_RATE = 0.40
    SCALE_FACTOR = 1.0

    embedding_dims = hp.get('embedding_dims')
    patch_size = hp.get('patch_size')
    contrastive_learning_patches = int(nr_of_timesteps * MASK_RATE / patch_size)
    masked_auto_encoder_patches = int(nr_of_timesteps / patch_size)

    learning_rate_callback = LearningRateCallback(
        d_model=embedding_dims,
        scale_factor=SCALE_FACTOR)

    masking_callback = MaskingCallback(
        nr_of_patches=masked_auto_encoder_patches,
        masking_rate=MASK_RATE,
        masks_feature='masks')

    masking_callback_cl = MaskingCallback(
        nr_of_patches=(masked_auto_encoder_patches - contrastive_learning_patches),
        masking_rate=MASK_RATE,
        masks_feature='cl_masks')

    my_callbacks = [
        learning_rate_callback,
        masking_callback,
        masking_callback_cl]

    return my_callbacks


class Tuner(kt.Hyperband):
    '''
    Hyperband based hyper-parameter tuning class
    '''
    def __init__(self, *args, nr_of_timesteps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nr_of_timesteps = nr_of_timesteps

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs['callbacks'] = get_callbacks(hp, self.nr_of_timesteps)
        return super().run_trial(trial, *args, **kwargs)

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        tf.keras.backend.clear_session()
        gc.collect()
