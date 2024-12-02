from data_loader import get_dataset
from utils import calculate_recall, Timer
from data_processor import partition_data
from hnsw import HNSW
import numpy as np

EF = 20
M = 16
K = 10
PARTITION_NUM = 10
THREAD_NUM = 16


def merge_top_k(result_lst: list, top_k: int) -> tuple:
    all_labels = np.concatenate([result[0] for result in result_lst], axis=1)
    all_distances = np.concatenate([result[1] for result in result_lst], axis=1)
    # Get the indicesof the sorted distances
    sort_indices = np.argsort(all_distances, axis=1)[:, :top_k]
    top_k_labels = np.take_along_axis(all_labels, sort_indices, axis=1)
    top_k_distances = np.take_along_axis(all_distances, sort_indices, axis=1)
    return top_k_labels, top_k_distances


def run_similar_partition():
    """
    Run the HNSW algorithm with similar partitioning strategy
    """
    # TODO: Implement the similar partitioning strategy
    # BUILD INDEX
    # 1. using k-means to partition the data
    # 2. sample data from each partition
    # 3. build a meta index with sampled data
    # 4. using graph partitioning to partition the data, assign each subgraph to a query node
    # 5. each query node build a index with the assigned partitions

    # QUERY
    # 1. find top-k partitions
    # 2. route to corresponding query node


def run_naive():
    """
    Run the HNSW algorithm with naive strategy
    """
    base, query, ground_truth = get_dataset("siftsmall")
    partitions = partition_data(base, PARTITION_NUM)
    timer = Timer()
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
    # run_similar_partition()
    run_naive()
