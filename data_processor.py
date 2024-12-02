import numpy as np


def partition_data(data: np.ndarray, num_partitions: int) -> list:
    print("Parition number: ", num_partitions)
    num, dim = data.shape
    partition_size = num // num_partitions
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        partitions.append(data[start:end])
    return partitions


def sample_data(data: np.ndarray, num_samples: int, method="random") -> np.ndarray:
    num, dim = data.shape
    indices = np.array([])
    if method == "random":
        indices = np.random.choice(num, num_samples, replace=False)
    elif method == "uniform":
        indices = np.linspace(0, num - 1, num_samples, dtype=int)
    else:
        raise ValueError("Invalid sampling method")
    return data[indices]
