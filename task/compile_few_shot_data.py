import argparse

import numpy as np

import os


def _get_args():
    parser = argparse.ArgumentParser(
        description="Compiling few-shot data"
    )

    parser.add_argument("--input_dir", type=str, default="test")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="test")

    return parser.parse_args()


if __name__ == '__main__':
    """
    compiles few-shot data
    """

    args = _get_args()

    input_dir = args.input_dir
    ratio = args.ratio
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    lb_train = np.load(f"{input_dir}/lb_train.npy")
    fc_train = np.load(f"{input_dir}/fc_train.npy")
    ts_train = np.load(f"{input_dir}/ts_train.npy")

    lb_val = np.load(f"{input_dir}/lb_val.npy")
    fc_val = np.load(f"{input_dir}/fc_val.npy")
    ts_val = np.load(f"{input_dir}/ts_val.npy")

    lb_test = np.load(f"{input_dir}/lb_test.npy")
    fc_test = np.load(f"{input_dir}/fc_test.npy")
    ts_test = np.load(f"{input_dir}/ts_test.npy")

    # take a random sample of training data
    num_train_samples = lb_train.shape[0]
    num_few_shot_samples = int(num_train_samples * ratio)
    few_shot_indices = np.random.choice(num_train_samples, num_few_shot_samples, replace=False)
    lb_train = lb_train[few_shot_indices]
    fc_train = fc_train[few_shot_indices]
    ts_train = ts_train[few_shot_indices]

    # take a random sample of validation data
    num_val_samples = lb_val.shape[0]
    num_few_shot_samples = int(num_val_samples * ratio)
    few_shot_indices = np.random.choice(num_val_samples, num_few_shot_samples, replace=False)
    lb_val = lb_val[few_shot_indices]
    fc_val = fc_val[few_shot_indices]
    ts_val = ts_val[few_shot_indices]

    # write before and after sample sizes
    print(f"Original training data size: {num_train_samples}")
    print(f"Few-shot training data size: {lb_train.shape[0]}")
    print(f"Original validation data size: {num_val_samples}")
    print(f"Few-shot validation data size: {lb_val.shape[0]}")

    np.save(os.path.join(output_dir, "lb_train.npy"), lb_train)
    np.save(os.path.join(output_dir, "fc_train.npy"), fc_train)
    np.save(os.path.join(output_dir, "ts_train.npy"), ts_train)
    np.save(os.path.join(output_dir, "lb_val.npy"), lb_val)
    np.save(os.path.join(output_dir, "fc_val.npy"), fc_val)
    np.save(os.path.join(output_dir, "ts_val.npy"), ts_val)
    np.save(os.path.join(output_dir, "lb_test.npy"), lb_test)
    np.save(os.path.join(output_dir, "fc_test.npy"), fc_test)
    np.save(os.path.join(output_dir, "ts_test.npy"), ts_test)

    print(f"Few-shot data compiled and saved to {output_dir}")
