import argparse
import sys
import numpy as np

import os

import tensorflow as tf

from tsf_model.models.pre_training import (
    build_model,
    compile_model,
    get_callbacks)

from utils import (
    get_metrics,
    predict_pre_train,
    read_json,
    save_json)


def _get_args():
    parser = argparse.ArgumentParser(description="Pre-train a foundation model")

    parser.add_argument("--input_dir", type=str, default="test")
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--nr_of_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--nr_of_seeds", type=int, default=1)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--l1_trend", type=float, default=0.01)
    parser.add_argument("--l2_trend", type=float, default=0.01)
    parser.add_argument("--l1_seasonality", type=float, default=0.01)
    parser.add_argument("--l2_seasonality", type=float, default=0.01)
    parser.add_argument("--l1_residual", type=float, default=0.01)
    parser.add_argument("--l2_residual", type=float, default=0.01)
    parser.add_argument("--nr_of_encoder_blocks", type=int, default=1)
    parser.add_argument("--nr_of_heads", type=int, default=1)
    parser.add_argument("--encoder_ffn_units", type=int, default=8)
    parser.add_argument("--embedding_dims", type=int, default=8)
    parser.add_argument("--projection_head_units", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.10)
    parser.add_argument("--mask_scalar", type=float, default=0.00)
    parser.add_argument("--mask_rate", type=float, default=0.40)
    parser.add_argument("--mae_threshold_comp", type=float, default=None)
    parser.add_argument("--mae_threshold_tre", type=float, default=None)
    parser.add_argument("--mae_threshold_sea", type=float, default=None)
    parser.add_argument("--cl_margin", type=float, default=0.25)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--prompt_pool_size", type=int, default=2)
    parser.add_argument("--nr_of_most_similar_prompts", type=int, default=1)
    parser.add_argument("--force_mae_comp", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--force_mae_tre", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--force_mae_sea", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--force_cl", type=int,
                        choices=[-1, 0, 1], default=0)
    parser.add_argument("--save_only_light_artifacts",
                        type=eval, default="False")
    parser.add_argument("--prefer_dense_to_time2vec",
                        type=eval, default="False")
    parser.add_argument("--w_comp", type=float, default=None)
    parser.add_argument("--w_tre", type=float, default=None)
    parser.add_argument("--w_sea", type=float, default=None)
    parser.add_argument("--w_cl", type=float, default=None)

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    return args


if __name__ == '__main__':
    '''Pre-trains a foundation model.'''

    # parse the args
    args = _get_args()

    input_dir = args.input_dir
    prompt_dir = args.prompt_dir
    output_dir = args.output_dir
    mask_rate = args.mask_rate
    mask_scalar = args.mask_scalar
    mini_batch_size = args.mini_batch_size
    clip_norm = args.clip_norm
    l1_trend = args.l1_trend
    l2_trend = args.l2_trend
    l1_seasonality = args.l1_seasonality
    l2_seasonality = args.l2_seasonality
    l1_residual = args.l1_residual
    l2_residual = args.l2_residual
    nr_of_encoder_blocks = args.nr_of_encoder_blocks
    nr_of_heads = args.nr_of_heads
    dropout_rate = args.dropout_rate
    encoder_ffn_units = args.encoder_ffn_units
    embedding_dims = args.embedding_dims
    projection_head_units = args.projection_head_units
    warmup_steps = args.warmup_steps
    scale_factor = args.scale_factor
    mae_threshold_comp = args.mae_threshold_comp
    mae_threshold_tre = args.mae_threshold_tre
    mae_threshold_sea = args.mae_threshold_sea
    cl_margin = args.cl_margin
    patch_size = args.patch_size
    prompt_pool_size = args.prompt_pool_size
    nr_of_most_similar_prompts = args.nr_of_most_similar_prompts
    patience = args.patience
    force_mae_comp = args.force_mae_comp
    force_mae_tre = args.force_mae_tre
    force_mae_sea = args.force_mae_sea
    force_cl = args.force_cl
    save_only_light_artifacts = args.save_only_light_artifacts
    prefer_dense_to_time2vec = args.prefer_dense_to_time2vec
    nr_of_seeds = args.nr_of_seeds
    w_comp = args.w_comp
    w_tre = args.w_tre
    w_sea = args.w_sea
    w_cl = args.w_cl

    # configure mixed precision policy
    tf.keras.mixed_precision.set_global_policy("float32")

    if embedding_dims % nr_of_heads != 0:
        raise ValueError(
            f'embedding_dims={embedding_dims} must be divisible '
            f'by nr_of_heads={nr_of_heads}. '
            f'Got remainder {embed_dim % nr_of_heads}.'
        )

    # read inputs.
    ds_train = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_train'))

    ds_val = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_validation'))

    ds_test = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_test'))

    if prompt_dir is not None:
        custom_prompt_keys_trend = read_json(
            directory=prompt_dir,
            file_name='prompts_trend.json')

        custom_prompt_keys_seasonality = read_json(
            directory=prompt_dir,
            file_name='prompts_seasonality.json')

        custom_prompt_keys_residual = read_json(
            directory=prompt_dir,
            file_name='prompts_residual.json')

        print(f'Trend prompts found: {len(custom_prompt_keys_trend)}')
        print(
            f'Seasonality prompts found: {len(custom_prompt_keys_seasonality)}'
            )
        print(f'Residual prompts found: {len(custom_prompt_keys_residual)}')
    else:
        custom_prompt_keys_trend = None
        custom_prompt_keys_residual = None
        custom_prompt_keys_seasonality = None
        print('No prompts found. Model is iniatilized with random prompts.')

    # batch datasets.
    ds_train = ds_train.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    # define paths
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.keras')

    # define patchs/covariates/
    use_time2vec = (ds_train.element_spec[-1].shape[1] > 0)
    nr_of_timesteps = ds_train.element_spec[0].shape[1]
    nr_of_covariates = ds_train.element_spec[0].shape[-1]

    contrastive_learning_patches = int(nr_of_timesteps * (mask_rate))
    contrastive_learning_patches = \
        int(contrastive_learning_patches / patch_size)

    best_loss = np.inf
    for seed in range(nr_of_seeds):
        print(f'Trial: {seed + 1} / {nr_of_seeds}')
        tf.keras.utils.set_random_seed(seed)

        # define model
        model = build_model(
            nr_of_covariates=nr_of_covariates,
            patch_size=patch_size,
            nr_of_encoder_blocks=nr_of_encoder_blocks,
            nr_of_heads=nr_of_heads,
            dropout_rate=dropout_rate,
            encoder_ffn_units=encoder_ffn_units,
            embedding_dims=embedding_dims,
            projection_head_units=projection_head_units,
            mask_rate=mask_rate,
            mask_scalar=mask_scalar,
            nr_of_timesteps=nr_of_timesteps,
            contrastive_learning_patches=contrastive_learning_patches,
            mae_threshold_comp=mae_threshold_comp,
            mae_threshold_tre=mae_threshold_tre,
            mae_threshold_sea=mae_threshold_sea,
            cl_margin=cl_margin,
            prompt_pool_size=prompt_pool_size,
            nr_of_most_similar_prompts=nr_of_most_similar_prompts,
            use_time2vec=use_time2vec,
            force_mae_comp=force_mae_comp,
            force_mae_tre=force_mae_tre,
            force_mae_sea=force_mae_sea,
            force_cl=force_cl,
            l1_trend=l1_trend,
            l2_trend=l2_trend,
            l1_seasonality=l1_seasonality,
            l2_seasonality=l2_seasonality,
            l1_residual=l1_residual,
            l2_residual=l2_residual,
            prefer_dense_to_time2vec=prefer_dense_to_time2vec,
            custom_prompt_keys_trend=custom_prompt_keys_trend,
            custom_prompt_keys_seasonality=custom_prompt_keys_seasonality,
            custom_prompt_keys_residual=custom_prompt_keys_residual,
            w_comp=w_comp,
            w_tre=w_tre,
            w_sea=w_sea,
            w_cl=w_cl)

        model = compile_model(model=model, clip_norm=clip_norm)

        callbacks = get_callbacks(
            embedding_dims=embedding_dims,
            warmup_steps=warmup_steps,
            scale_factor=scale_factor,
            patience=patience,
            mae_threshold_comp=mae_threshold_comp,
            mae_threshold_tre=mae_threshold_tre,
            mae_threshold_sea=mae_threshold_sea)

        # fit the model
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=3,
            verbose=2,
            shuffle=False,
            callbacks=callbacks)

        val_loss = history.history['val_mae_composed'][-1]

        if val_loss < best_loss:
            best_loss = val_loss
            model.save(checkpoint_path)

    # Load the best initialization
    model = tf.keras.models.load_model(checkpoint_path)
    model = compile_model(model=model, clip_norm=clip_norm)

    # define callbacks
    callbacks = get_callbacks(
        embedding_dims=embedding_dims,
        warmup_steps=warmup_steps,
        scale_factor=scale_factor,
        patience=patience,
        mae_threshold_comp=mae_threshold_comp,
        mae_threshold_tre=mae_threshold_tre,
        mae_threshold_sea=mae_threshold_sea)

    # fit the model
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.nr_of_epochs,
        verbose=2,
        shuffle=False,
        callbacks=callbacks)

    # predict
    actual_train, pred_train, mask_train = predict_pre_train(
        model=model,
        ds=ds_train)
    actual_val, pred_val, mask_val = predict_pre_train(
        model=model,
        ds=ds_val)
    actual_test, pred_test, mask_test = predict_pre_train(
        model=model,
        ds=ds_test)

    # calculate metrics
    metrics = get_metrics(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        history=history)

    # save outputs
    save_json(metrics, output_dir, 'metrics.json')
    save_json(vars(args), output_dir, 'params.json')
    save_json(history.history, output_dir, 'history.json')

    if not save_only_light_artifacts:
        model.save(os.path.join(output_dir, 'model.keras'))

        actual_train.save(os.path.join(output_dir, 'actual_train'))
        pred_train.save(os.path.join(output_dir, 'pred_train'))
        mask_train.save(os.path.join(output_dir, 'mask_train'))
        actual_val.save(os.path.join(output_dir, 'actual_val'))
        pred_val.save(os.path.join(output_dir, 'pred_val'))
        mask_val.save(os.path.join(output_dir, 'mask_val'))
        actual_test.save(os.path.join(output_dir, 'actual_test'))
        pred_test.save(os.path.join(output_dir, 'pred_test'))
        mask_test.save(os.path.join(output_dir, 'mask_test'))

    os.remove(checkpoint_path)

    print('Successful.')
