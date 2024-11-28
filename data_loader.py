import numpy as np


def read_ivecs(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def read_fvecs(fname):
    return read_ivecs(fname).view("float32")
