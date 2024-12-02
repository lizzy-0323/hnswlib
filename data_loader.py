import numpy as np


def get_dataset(dataset: str) -> tuple:
    if dataset == "sift":
        return (
            load_data("./data/sift/sift_base.fvecs"),
            load_data("./data/sift/sift_query.fvecs"),
            load_data("./data/sift/sift_groundtruth.ivecs", "ivecs"),
        )
    if dataset == "siftsmall":
        return (
            load_data("./data/siftsmall/siftsmall_base.fvecs"),
            load_data("./data/siftsmall/siftsmall_query.fvecs"),
            load_data("./data/siftsmall/siftsmall_groundtruth.ivecs", "ivecs"),
        )
    raise ValueError("Invalid dataset")


def load_data(file_name: str, file_type: str = "fvecs") -> np.ndarray:
    if file_type == "fvecs":
        return read_fvecs(file_name)
    if file_type == "ivecs":
        return read_ivecs(file_name)
    raise ValueError("Invalid file type")


def read_ivecs(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def read_fvecs(fname):
    return read_ivecs(fname).view("float32")
