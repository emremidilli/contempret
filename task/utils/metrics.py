import tensorflow as tf


def get_metrics(
        model: tf.keras.Model,
        ds_train: tf.data.Dataset,
        ds_val: tf.data.Dataset,
        ds_test: tf.data.Dataset,
        history: tf.keras.callbacks.History) -> dict:
    '''calculates metrics'''

    # calculate metrics
    train_evaluation = model.evaluate(ds_train, return_dict=True)
    validation_evaluation = model.evaluate(ds_val, return_dict=True)
    test_evaluation = model.evaluate(ds_test, return_dict=True)
    trainable_params = tf.reduce_sum([
        tf.reduce_prod(var.shape)
        for var in model.trainable_variables])

    non_trainable_params = tf.reduce_sum([
        tf.reduce_prod(var.shape)
        for var in model.non_trainable_variables])

    # Create the metrics dictionary
    metrics = {
        'train_evaluation': train_evaluation,
        'validation_evaluation': validation_evaluation,
        'test_evaluation': test_evaluation,
        'trainable_params': int(trainable_params.numpy()),
        'non_trainable_params': int(non_trainable_params.numpy()),
        'epochs': history.epoch,
    }

    return metrics
