import tensorflow as tf

from utils import (
    get_args,
    save_json)


if __name__ == '__main__':
    '''
    Makes inference for batch prediction.
    It requires input dataset to have both input features
    as well as ground truth.
    '''
    args = get_args()
    print(args)
    input_dir = args.input_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    mini_batch_size = args.mini_batch_size

    # get inputs
    ds = tf.data.Dataset.load(input_dir)
    model = tf.keras.models.load_model(model_dir, compile=True)

    # batch datasets
    ds = ds.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    # predict
    evaluation_results = model.evaluate(ds, return_dict=True)

    # save outputs
    save_json(evaluation_results, output_dir, 'metrics.json')
