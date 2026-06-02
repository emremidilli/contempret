import argparse
import os
import sys

import tensorflow as tf

from tsf_model.models.fine_tuning import SequencePredictor  # needed for custom Keras object deserialization
from utils import predict_fine_tune, save_json


def _get_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")

    parser.add_argument("--input_dir", type=str, default="test")
    parser.add_argument("--model_dir", type=str, default="test")
    parser.add_argument("--foundation_model_dir", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--mini_batch_size", type=int, default=128)

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    return args


if __name__ == '__main__':
    '''
    Makes inference for batch prediction.
    It requires input dataset to have both input features
    as well as ground truth.
    '''
    args = _get_args()
    print(args)
    input_dir = args.input_dir
    model_dir = args.model_dir
    foundation_model_dir = args.foundation_model_dir
    output_dir = args.output_dir
    mini_batch_size = args.mini_batch_size

    # configure mixed precision policy
    tf.keras.mixed_precision.set_global_policy("float32")

    # get inputs
    ds = tf.data.Dataset.load(input_dir)
    foundation_model = tf.keras.models.load_model(
        os.path.join(foundation_model_dir, 'model.keras'),
        compile=False)
    model = tf.keras.models.load_model(
        os.path.join(model_dir, 'model.keras'),
        compile=False)

    # batch datasets
    ds = ds.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    # predict
    y_truth, y_pred = predict_fine_tune(
        model=model,
        ds=ds)

    mae_fn = tf.keras.metrics.MeanAbsoluteError(name='mae')
    cos_fn = tf.keras.metrics.CosineSimilarity(name='cos')
    mse_fn = tf.keras.metrics.MeanSquaredError(name='mse')

    metrics = dict()
    metrics["mae"] = mae_fn(y_truth, y_pred).numpy().item()
    metrics["mse"] = mse_fn(y_truth, y_pred).numpy().item()
    metrics["cos"] = cos_fn(y_truth, y_pred).numpy().item()

    # save outputs
    save_json(metrics, output_dir, 'metrics.json')
