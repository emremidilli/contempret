import tensorflow as tf

from typing import Tuple


def _unbatch_dataset_pt(
        ds: tf.data.Dataset):
    '''
    covert tf.data.Dataset to single batch tensors.
    it applies to only pre-training dataset.
    '''

    # Reverting back to separate tensors
    tre_list, sea_list, res_list, ts_list = [], [], [], []

    for tre_batch, sea_batch, res_batch, ts_batch in ds:
        tre_list.append(tre_batch)
        sea_list.append(sea_batch)
        res_list.append(res_batch)
        ts_list.append(ts_batch)

    # Concatenate the lists to form tensors
    tre = tf.concat(tre_list, axis=0)
    sea = tf.concat(sea_list, axis=0)
    res = tf.concat(res_list, axis=0)
    ts = tf.concat(ts_list, axis=0)

    new_ds = tf.data.Dataset.from_tensor_slices(
        (tre, sea, res, ts))

    return new_ds


def predict_pre_train(
        model: tf.keras.Model,
        ds: tf.data.Dataset) -> Tuple[
            tf.data.Dataset,
            tf.data.Dataset,
            tf.data.Dataset]:
    '''
    predicts an input dataset.
    it applies to pre-training model.
    '''
    ds = _unbatch_dataset_pt(ds)

    npy_input = list(ds.batch(len(ds)).as_numpy_iterator())[0]

    pred_tre, pred_sea, pred_res, _ = \
        model.predict(npy_input)

    pred = tf.data.Dataset.from_tensor_slices((pred_tre, pred_sea, pred_res))

    mask = tf.data.Dataset.from_tensor_slices(model.masks)

    return ds, pred, mask


def _unbatch_dataset_ft(
        ds: tf.data.Dataset):
    '''
    covert tf.data.Dataset to single batch tensors.
    it applies only to fine-tuning dataset.
    '''

    # Reverting back to separate tensors
    tre_list, sea_list, res_list, ts_list, lbl_list = [], [], [], [], []

    for (tre_batch, sea_batch, res_batch, ts_batch), lbl_batch in ds:
        tre_list.append(tre_batch)
        sea_list.append(sea_batch)
        res_list.append(res_batch)
        ts_list.append(ts_batch)
        lbl_list.append(lbl_batch)

    # Concatenate the lists to form tensors
    tre = tf.concat(tre_list, axis=0)
    sea = tf.concat(sea_list, axis=0)
    res = tf.concat(res_list, axis=0)
    ts = tf.concat(ts_list, axis=0)
    lbl = tf.concat(lbl_list, axis=0)

    new_input = tf.data.Dataset.from_tensor_slices(
        (tre, sea, res, ts))

    new_lbl = tf.data.Dataset.from_tensor_slices(
        (lbl))

    return new_input, new_lbl


def predict_fine_tune(
        model: tf.keras.Model,
        ds: tf.data.Dataset) -> Tuple[
            tf.data.Dataset,
            tf.data.Dataset]:
    '''
    predicts an input dataset and saves
    the input and prediction into tmp folder.
    it applies to fine-tuning model.
    '''
    ds_input, ds_lbl = _unbatch_dataset_ft(ds)

    npy_input = list(ds_input.batch(len(ds_input)).as_numpy_iterator())[0]
    npy_lbl = list(ds_lbl.batch(len(ds_lbl)).as_numpy_iterator())[0]

    npy_pred = model.predict(npy_input)

    return npy_lbl, npy_pred
