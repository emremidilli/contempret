import gc

import keras_tuner as kt

import tensorflow as tf

from tsf_model.models.pre_training import (
    build_model,
    compile_model)

from utils import LearningRateCallback


def build_model_to_tune(
    hp: kt.HyperParameters,
    nr_of_timesteps: int,
    nr_of_covariates: int,
    use_time2vec: bool,
    training_mode: str):
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
    CLIP_NORM = 0.10

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
    projection_head_units = \
        hp.Choice('projection_head_units', values=[8, 16, 32, 64])
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

    w_comp = None
    w_tre = None
    w_sea = None
    w_cl = None
    if training_mode == "weighted":
        w_comp = hp.Float(
            'w_comp',
            min_value=0.01,
            max_value=1.00,
            sampling='log')

        w_tre = hp.Float(
            'w_tre',
            min_value=0.01,
            max_value=1.00,
            sampling='log')

        w_sea = hp.Float(
            'w_sea',
            min_value=0.01,
            max_value=1.00,
            sampling='log')

        w_cl = hp.Float(
            'w_cl',
            min_value=0.01,
            max_value=1.00,
            sampling='log')

    # define model
    contrastive_learning_patches = int(nr_of_timesteps * (MASK_RATE))
    contrastive_learning_patches = \
        int(contrastive_learning_patches / patch_size)

    model = build_model(
        nr_of_covariates=nr_of_covariates,
        patch_size=patch_size,
        nr_of_encoder_blocks=nr_of_encoder_blocks,
        nr_of_heads=nr_of_heads,
        dropout_rate=dropout_rate,
        encoder_ffn_units=encoder_ffn_units,
        embedding_dims=embedding_dims,
        projection_head_units=projection_head_units,
        mask_rate=MASK_RATE,
        mask_scalar=MASK_SCALAR,
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
        custom_prompt_keys_residual=CUSTOM_PROMPT_KEYS_RESIDUAL,
        w_comp=w_comp,
        w_tre=w_tre,
        w_sea=w_sea,
        w_cl=w_cl)

    model = compile_model(model=model, clip_norm=CLIP_NORM)

    return model


def get_callbacks(hp):
    '''
    returns the callbacks for hyperparameter tuning.
    '''
    SCALE_FACTOR = 1.0

    embedding_dims = hp.get('embedding_dims')

    learning_rate_callback = LearningRateCallback(
        d_model=embedding_dims,
        scale_factor=SCALE_FACTOR)

    my_callbacks = [
        learning_rate_callback]

    return my_callbacks


class Tuner(kt.Hyperband):
    '''
    Hyperband based hyper-parameter tuning class
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        kwargs['callbacks'] = get_callbacks(hp)
        return super().run_trial(trial, *args, **kwargs)

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        tf.keras.backend.clear_session()
        gc.collect()
