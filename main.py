from data_loader import read_ivecs, read_fvecs
from utils import calculate_recall
import time
import numpy as np
import hnswlib

EF = 20
M = 16
K = 10
CLIENT_NUM = 4
THREAD_NUM = 16


class Timer:
    """
    Timer class to caculate time
    """

    def __init__(self):
        self.start = time.time()

    def tick(self):
        self.start = time.time()

    def tuck(self, message: str = ""):
        # caculate time in seconds
        print(f"{message} time: {time.time() - self.start:.2f}s")


class HNSW:
    def __init__(
        self,
        dim: int,
        max_elements: int,
        ef_construction: int,
        thread_num: int,
        M: int,
        index: str,
        index_type: str = "hnsw",
        nth:int = 0,
    ):
        self.offset = nth * max_elements
        if index_type == "hnsw":
            # init index
            self.index = hnswlib.Index(space="l2", dim=dim)
            # if os.path.exists(index):
            #     self.load_index(index)
            # else:
            self.index.init_index(
                max_elements=max_elements, ef_construction=ef_construction, M=M
            )
            self.index.set_ef(EF)
            self.index.set_num_threads(thread_num)
        else:
            self.index = hnswlib.BFIndex(space="l2", dim=dim)
            self.index.init_index(max_elements=max_elements)

    def add_items(self, data: np.ndarray):
        self.index.add_items(data)

    def query(self, query: np.ndarray, k: int):
        return self.index.knn_query(query, k)

    def save_index(self, path: str):
        self.index.save_index(path)

    def load_index(self, path: str):
        self.index.load_index(path)


def partition_data(data: np.ndarray, num_partitions: int) -> list:
    num, dim = data.shape
    partition_size = num // num_partitions
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        partitions.append(data[start:end])
    return partitions


def merge_top_k(result_lst: list, top_k: int) -> tuple:
    all_labels = np.concatenate([result[0] for result in result_lst], axis=1)
    all_distances = np.concatenate([result[1] for result in result_lst], axis=1)
    # Get the indicesof the sorted distances
    sort_indices = np.argsort(all_distances, axis=1)[:, :top_k]
    top_k_labels = np.take_along_axis(all_labels, sort_indices, axis=1)
    top_k_distances = np.take_along_axis(all_distances, sort_indices, axis=1)
    return top_k_labels, top_k_distances


def load_data(file_name: str, file_type: str = "fvecs") -> np.ndarray:
    if file_type == "fvecs":
        return read_fvecs(file_name)
    if file_type == "ivecs":
        return read_ivecs(file_name)
    raise ValueError("Invalid file type")


def run():
    # base = load_data("data/siftsmall/siftsmall_base.fvecs")
    # query = load_data("data/siftsmall/siftsmall_query.fvecs")
    # ground_truth = load_data("data/siftsmall/siftsmall_groundtruth.ivecs", "ivecs")

    base = load_data("data/sift/sift_base.fvecs")
    query = load_data("data/sift/sift_query.fvecs")
    ground_truth = load_data("data/sift/sift_groundtruth.ivecs", "ivecs")
    partitions = partition_data(base, CLIENT_NUM)
    timer = Timer()
    print("Num partitions:", len(partitions))
    print(partitions[0].shape)
    # init hnsw
    timer.tick()
    hnsw_lst = [
        HNSW(
            dim=partition.shape[1],
            max_elements=partition.shape[0],
            ef_construction=200,
            thread_num=THREAD_NUM,
            M=M,
            index="index" + str(i) + ".bin",
            # index_type="bf",
            nth=i,
        )
        for i, partition in enumerate(partitions)
    ]
    for hnsw, partition in zip(hnsw_lst, partitions):
        hnsw.add_items(partition)
    timer.tuck("Index build")

    # query
    timer.tick()
    # result_lst = [hnsw.query(query, 100) for hnsw in hnsw_lst]
    result_lst = []
    for hnsw in hnsw_lst:
        relative_labels, distances = hnsw.query(query, 100)
        labels = relative_labels + hnsw.offset
        result_lst.append((labels, distances))
    timer.tuck("Query")

    # merge top k
    timer.tick()
    labels, distances = merge_top_k(result_lst, K)
    timer.tuck("Aggregate")

    # calculate recall
    recall = calculate_recall(ground_truth, labels, K)
    print(f"Recall: {recall}")


if __name__ == "__main__":
    run()
