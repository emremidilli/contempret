import argparse
import json
import sys

import keras_tuner as kt

import os

import tensorflow as tf

from tsf_model.tuner.tuner import build_model_to_tune, Tuner


def _get_args():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters of a foundation model")

    parser.add_argument("--input_dir", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--executions_per_trial", type=int, default=1)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument(
        "--training_mode",
        choices=["sequential", "weighted"],
        default="sequential")

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    return args


if __name__ == '__main__':
    '''Tunes the hyperparameters of foundation model.'''
    # parse the args
    args = _get_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    mini_batch_size = args.mini_batch_size
    max_epochs = args.max_epochs
    executions_per_trial = args.executions_per_trial
    factor = args.factor
    training_mode = args.training_mode

    # set constants
    HP_PROJECT = 'my_project'

    # configure mixed precision policy
    tf.keras.mixed_precision.set_global_policy("float32")

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
        hypermodel=lambda hp: build_model_to_tune(
            hp,
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            use_time2vec=use_time2vec,
            training_mode=training_mode
        ),
        objective=kt.Objective('val_mae_composed', direction='min'),
        max_epochs=max_epochs,
        factor=factor,
        executions_per_trial=executions_per_trial,
        directory=output_dir,
        project_name=HP_PROJECT)

    tuner.search(
        ds_train,
        validation_data=ds_val)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0].values
    best_hp_path = os.path.join(output_dir, 'best_hyperparameters.json')
    with open(best_hp_path, 'w') as f:
        json.dump(best_hp, f, indent=2)

    print('Successful.')
