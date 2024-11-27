import os
from data_loader import fvecs_read, ivecs_read
from utils import calculate_recall
import time
import numpy as np
import hnswlib

EF_CONSTRUCTION = 40
M = 16
K = 1
CLIENT_NUM = 4
THREAD_NUM = 4


class HNSW:
    def __init__(
        self,
        dim: int,
        max_elements: int,
        ef_construction: int,
        thread_num: int,
        M: int,
        index: str,
        save_index: bool = False,
    ):
        # init index
        self.index = hnswlib.Index(space="l2", dim=dim)
        # if os.path.exists(index):
        #     self.load_index(index)
        # else:
        self.index.init_index(
            max_elements=max_elements, ef_construction=ef_construction, M=M
        )
        self.index.set_ef(ef_construction)
        self.index.set_num_threads(thread_num)
        if save_index:
            self.save_index(index)

    def add_items(self, data: np.ndarray):
        self.index.add_items(data)

    def query(self, query: np.ndarray, k: int):
        return self.index.knn_query(query, k)

    def save_index(self, path: str):
        self.index.save_index(path)

    def load_index(self, path: str):
        self.index.load_index(path)


class Timer:
    def __init__(self):
        self.start = time.time()

    def tick(self):
        self.start = time.time()

    def tuck(self):
        # caculate time in seconds
        print(f"Time: {time.time() - self.start:.2f}s")


def partition_data(data: np.ndarray, num_partitions: int) -> list:
    num, dim = data.shape
    partition_size = num // num_partitions
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        partitions.append(data[start:end])
    return partitions


def merge_top_k(*result_lst, top_k: int) -> tuple:
    all_labels = np.concatenate([result[0] for result in result_lst], axis=1)
    all_distances = np.concatenate([result[1] for result in result_lst], axis=1)
    # Get the indicesof the sorted distances
    sort_indices = np.argsort(all_distances, axis=1)
    top_k_labels = all_labels[
        np.arange(all_labels.shape[0])[:, None], sort_indices[:, :top_k]
    ]
    top_k_distances = all_distances[
        np.arange(all_distances.shape[0])[:, None], sort_indices[:, :top_k]
    ]

    return top_k_labels, top_k_distances


def load_data(file_name: str, file_type: str = "fvecs") -> np.ndarray:
    if file_type == "fvecs":
        return fvecs_read(file_name)
    if file_type == "ivecs":
        return ivecs_read(file_name)
    raise ValueError("Invalid file type")


def run():
    base = load_data("data/siftsmall/siftsmall_base.fvecs")
    query = load_data("data/siftsmall/siftsmall_query.fvecs")
    ground_truth = load_data("data/siftsmall/siftsmall_groundtruth.ivecs", "ivecs")

    # base = load_data("data/sift/sift_base.fvecs")
    # query = load_data("data/sift/sift_query.fvecs")
    # ground_truth = load_data("data/sift/sift_groundtruth.ivecs", "ivecs")
    partitions = partition_data(base, CLIENT_NUM)
    print("Num partitions:", len(partitions))
    hnsw_lst = [
        HNSW(
            dim=partition.shape[1],
            max_elements=partition.shape[0],
            ef_construction=EF_CONSTRUCTION,
            thread_num=THREAD_NUM,
            M=M,
            index="index" + str(i) + ".bin",
            save_index=True,
        )
        for i, partition in enumerate(partitions)
    ]
    for hnsw, partition in zip(hnsw_lst, partitions):
        hnsw.add_items(partition)
    result_lst = [hnsw.query(query, 20) for hnsw in hnsw_lst]
    # merge top k
    labels, distances = merge_top_k(*result_lst, top_k=K)
    recall = calculate_recall(ground_truth, labels, top_k=K)
    print(f"Recall: {recall}")

    timer = Timer()
    # query
    timer.tick()
    # result = [labels, distances]
    timer.tuck()


if __name__ == "__main__":
    run()
