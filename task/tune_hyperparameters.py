import keras_tuner as kt

import os

import tensorflow as tf

from tsf_model import build_model, Tuner

from utils import (
    get_args)

if __name__ == '__main__':
    '''Tunes the hyperparameters of foundation model.'''
    # parse the args
    args = get_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    mini_batch_size = args.mini_batch_size
    max_epochs = args.max_epochs
    executions_per_trial = args.executions_per_trial
    factor = args.factor

    # set constants
    HP_PROJECT = 'my_project'

    # get inputs.
    ds_train = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_train'))

    ds_val = tf.data.Dataset.load(
        path=os.path.join(input_dir, 'dataset_validation'))

    # batch datasets.
    ds_train = ds_train\
        .shuffle(buffer_size=10000)\
        .take(1024)\
        .batch(mini_batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val\
        .shuffle(buffer_size=10000)\
        .take(1024)\
        .batch(mini_batch_size)\
        .prefetch(tf.data.AUTOTUNE)

    use_time2vec = (ds_train.element_spec[-1].shape[1] > 0)
    nr_of_timesteps = ds_train.element_spec[0].shape[1]
    nr_of_covariates = ds_train.element_spec[0].shape[-1]

    tuner = Tuner(
        hypermodel=lambda hp: build_model(
            hp,
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            use_time2vec=use_time2vec
        ),
        objective=kt.Objective('val_mae_composed', direction='min'),
        max_epochs=max_epochs,
        factor=factor,
        executions_per_trial=executions_per_trial,
        directory=output_dir,
        project_name=HP_PROJECT,
        nr_of_timesteps=nr_of_timesteps)

    tuner.search(
        ds_train,
        validation_data=ds_val)

    print('Successful.')
