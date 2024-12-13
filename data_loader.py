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


def read_bvecs(fname):
    a = np.fromfile(fname, dtype="uint8")
    d = a.view("int32")[0]
    return a.reshape(-1, d + 4)[:, 4:].copy()


def read_fbin(fname):
    shape = np.fromfile(fname, dtype=np.uint32, count=2)
    if float(shape[0]) * shape[1] * 4 > 2_000_000_000:
        data = np.memmap(fname, dtype=np.float32, offset=8, mode="r").reshape(shape)
    else:
        data = np.fromfile(fname, dtype=np.float32, offset=8).reshape(shape)
    return data
