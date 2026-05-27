import argparse
import numpy as np
import sys

import os

import tensorflow as tf

from tsf_model.models.pre_processing import InputPreProcessor

from utils import read_npy


def _get_args():
    parser = argparse.ArgumentParser(
        description="Pre-process datasets for fine-tuning")

    parser.add_argument("--input_dir", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--pool_size_trend", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=1.96)

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    return args


if __name__ == '__main__':
    '''
    Converts formatted datasets to tf.data.Dataset format
        for fine-tuning process.
    Saves final dataset and pre-processor.
    '''
    args = _get_args()
    print(args)
    input_dir = args.input_dir
    output_dir = args.output_dir
    pool_size_trend = args.pool_size_trend
    sigma = args.sigma

    # read inputs
    lb_train = read_npy(directory=input_dir, file_name='lb_train.npy')
    lb_val = read_npy(directory=input_dir, file_name='lb_val.npy')
    lb_test = read_npy(directory=input_dir, file_name='lb_test.npy')

    fc_train = read_npy(directory=input_dir, file_name='fc_train.npy')
    fc_val = read_npy(directory=input_dir, file_name='fc_val.npy')
    fc_test = read_npy(directory=input_dir, file_name='fc_test.npy')

    ts_train = read_npy(directory=input_dir, file_name='ts_train.npy')
    ts_val = read_npy(directory=input_dir, file_name='ts_val.npy')
    ts_test = read_npy(directory=input_dir, file_name='ts_test.npy')

    # build datasets
    use_timestamp = (ts_train is not None)

    nr_of_covariates = lb_train.shape[-1]
    input_pre_processor = InputPreProcessor(
        pool_size_trend=pool_size_trend,
        nr_of_covariates=nr_of_covariates,
        sigma=sigma,
        scale_data=False,
        use_timestamp=use_timestamp)

    lb_train_val = np.concatenate((lb_train, lb_val), axis=0)
    if use_timestamp:
        ts_train_val = np.concatenate((ts_train, ts_val), axis=0)

        input_train = (lb_train, ts_train)
        input_val = (lb_val, ts_val)
        input_test = (lb_test, ts_test)
        input_train_val = (lb_train_val, ts_train_val)
    else:
        input_train = (lb_train, )
        input_val = (lb_val, )
        input_test = (lb_test, )
        input_train_val = (lb_train_val, )

    # process
    lb_tre_train, lb_sea_train, lb_res_train, ts_train = \
        input_pre_processor(input_train)
    lb_tre_val, lb_sea_val, lb_res_val, ts_val = \
        input_pre_processor(input_val)
    lb_tre_test, lb_sea_test, lb_res_test, ts_test = \
        input_pre_processor(input_test)

    ds_train = tf.data.Dataset.from_tensor_slices(
        ((lb_tre_train, lb_sea_train, lb_res_train, ts_train), fc_train))

    ds_val = tf.data.Dataset.from_tensor_slices(
        ((lb_tre_val, lb_sea_val, lb_res_val, ts_val), fc_val))

    ds_test = tf.data.Dataset.from_tensor_slices(
        ((lb_tre_test, lb_sea_test, lb_res_test, ts_test), fc_test))

    # save outputs
    os.makedirs(output_dir, exist_ok=True)

    ds_train.save(
        os.path.join(output_dir, 'dataset_train'))

    ds_val.save(
        os.path.join(output_dir, 'dataset_validation'))

    ds_test.save(
        os.path.join(output_dir, 'dataset_test'))

    input_pre_processor.save(
        os.path.join(output_dir, 'input_preprocessor'),
        overwrite=True,
        save_format='tf')

    print ('Successful.')
