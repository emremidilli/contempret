import numpy as np

import os

import tensorflow as tf

from tsf_model import (
    SequencePredictor,
    TimeSeriesClassifier)

from utils import (
    get_args,
    get_metrics,
    predict_fine_tune,
    RamCleaner,
    save_json,
    save_npy)


if __name__ == '__main__':
    '''Fine tunes a pre-trained model.'''
    # parse args
    args = get_args()
    print(args)

    input_dir = args.input_dir
    output_dir = args.output_dir
    pre_trained_model_dir = args.pre_trained_model_dir
    mini_batch_size = args.mini_batch_size
    learning_rate = args.learning_rate
    clip_norm = args.clip_norm
    nr_of_epochs = args.nr_of_epochs
    patience = args.patience
    warmup_epochs_early_stopping = args.warmup_epochs_early_stopping
    task_type = args.task_type
    tune_time2vec = args.tune_time2vec
    nr_of_seeds = args.nr_of_seeds

    # get inputs
    ds_train = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_train'))

    ds_val = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_validation'))

    ds_test = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_test'))

    pre_trained_model = tf.keras.models.load_model(
        os.path.join(pre_trained_model_dir, 'saved_model'))

    # batch datasets
    ds_train = ds_train.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    # build model
    try:
        trend_prompt = pre_trained_model.trend_prompt
        seasonality_prompt = pre_trained_model.seasonality_prompt
        residual_prompt = pre_trained_model.residual_prompt
    except:
        trend_prompt = None
        seasonality_prompt = None
        residual_prompt = None

    if task_type == 'sequence_prediction':
        univariate_model_type = SequencePredictor
        loss_fn = tf.keras.losses.MeanSquaredError(name='mse')
        metrics_fn = [
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.CosineSimilarity(name='cos'),
            tf.keras.metrics.MeanSquaredError(name='mse')
            ]
        monitor = 'val_mae'
    elif task_type == 'time_series_classification':
        univariate_model_type = TimeSeriesClassifier
        loss_fn = tf.keras.losses.BinaryCrossentropy (name='bce')
        metrics_fn = [
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
            ]
        monitor = 'val_loss'

    best_model_weights = None
    best_loss = np.inf
    for seed in range(nr_of_seeds):
        print(f'Trial: {seed + 1} / {nr_of_seeds}')
        tf.keras.utils.set_random_seed(seed)

        # define optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clip_norm)

        model = univariate_model_type(
            patch_tokenizer=pre_trained_model.patch_tokenizer,
            encoder_representation=pre_trained_model.encoder_representation,
            nr_of_timesteps=ds_train.element_spec[1].shape[1],
            trend_prompt=trend_prompt,
            seasonality_prompt=seasonality_prompt,
            residual_prompt=residual_prompt,
            revIn_tre=pre_trained_model.revIn_tre,
            revIn_sea=pre_trained_model.revIn_sea,
            revIn_res=pre_trained_model.revIn_res,
            tre_embedding=pre_trained_model.tre_embedding,
            sea_embedding=pre_trained_model.sea_embedding,
            res_embedding=pre_trained_model.res_embedding,
            tune_time2vec=tune_time2vec)

        model.compile(
            run_eagerly=False,
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics_fn)

        # fit model (briefly)
        history = model.fit(
            ds_train,
            epochs=5,
            verbose=2,
            validation_data=ds_val,
            shuffle=False)

        val_loss = history.history['val_loss'][-1]

        if val_loss < best_loss:
            best_loss = val_loss
             # store weights
            best_model_weights = model.get_weights()

    # Load the best initialization
    model.set_weights(best_model_weights)

    # define callbacks
    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    ram_cleaner_callback = RamCleaner()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        start_from_epoch=warmup_epochs_early_stopping,
        restore_best_weights=True)

    callbacks = [
        terminate_on_nan_callback,
        ram_cleaner_callback,
        early_stopping]

    # fit the best initialization
    history = model.fit(
        ds_train,
        epochs=nr_of_epochs,
        verbose=2,
        validation_data=ds_val,
        shuffle=False,
        callbacks=callbacks)

    # predict
    actual_train, pred_train = predict_fine_tune(
        model=model,
        ds=ds_train)
    actual_val, pred_val = predict_fine_tune(
        model=model,
        ds=ds_val)
    actual_test, pred_test = predict_fine_tune(
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
    os.makedirs(output_dir, exist_ok=True)
    model.save(
        os.path.join(output_dir, 'saved_model'),
        overwrite=True,
        save_format='tf')

    save_json(metrics, output_dir, 'metrics.json')
    save_json(vars(args), output_dir, 'params.json')
    save_json(history.history, output_dir, 'history.json')

    save_npy(actual_train, output_dir,'actual_train.npy')
    save_npy(pred_train, output_dir,'pred_train.npy')
    save_npy(actual_val, output_dir,'actual_val.npy')
    save_npy(pred_val, output_dir,'pred_val.npy')
    save_npy(actual_test, output_dir,'actual_test.npy')
    save_npy(pred_test, output_dir,'pred_test.npy')

    print('Successful.')
