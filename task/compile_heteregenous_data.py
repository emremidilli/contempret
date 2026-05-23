import argparse

import numpy as np

import os


def _get_args():
    parser = argparse.ArgumentParser(
        description="Compiling heteregenous data"
    )

    parser.add_argument("--input_dir_parent", type=str, default="test")
    parser.add_argument("--children_list", type=eval, default="['test']")
    parser.add_argument("--output_dir", type=str, default="test")

    return parser.parse_args()


if __name__ == '__main__':
    """
    concats multiple datasets with each other.
    """

    args = _get_args()

    input_dir_parent = args.input_dir_parent
    children_list = args.children_list
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    first_child_dir = os.path.join(input_dir_parent, children_list[0])
    npy_files = [f for f in os.listdir(first_child_dir) if f.endswith(".npy")]

    for fname in npy_files:
        arrays = []
        for child in children_list:
            fpath = os.path.join(input_dir_parent, child, fname)
            arrays.append(np.load(fpath))
        shapes = [a.shape for a in arrays]
        for i, shape in enumerate(shapes[1:], 1):
            if shape[1:] != shapes[0][1:]:
                raise ValueError(
                    f"{fname}: shape mismatch at child '{children_list[i]}': "
                    f"expected {shapes[0][1:]} (axes 1+), got {shape[1:]}"
                )
        concatenated = np.concatenate(arrays, axis=0)
        np.save(os.path.join(output_dir, fname), concatenated)
        print(f"Saved {fname}: shape {concatenated.shape}")
