from typing import Literal
import time
from joblib import dump
import data_processor
import numpy as np
from hnsw import HNSW
from data_loader import get_dataset
from data_processor import (
    build_cluster,
    kmeans_partition,
    partition_buckets_spectral,
    partition_buckets_kmeans,
    build_nodes,
)
from utils import Timer
from node import Node
import utils
import pickle

timer = Timer()


def save_persistent_cluster(cluster, dataset_name):
    """
    Create a persistent cluster
    """
    with open(f"./data/pickle/{dataset_name}_cluster.pkl", "wb") as f:
        pickle.dump(cluster, f)


def create_persistent_cluster(
    data, num_buckets, num_nodes, partition_method, dataset_name
):
    """
    Create a persistent cluster
    """
    cluster = build_cluster(data, num_buckets, num_nodes, partition_method)
    save_persistent_cluster(cluster, dataset_name)


def load_persistent_cluster(dataset_name):
    """
    Load a persistent cluster
    """
    with open(f"./data/pickle/{dataset_name}_cluster.pkl", "rb") as f:
        cluster = pickle.load(f)
    return cluster


def distributed_search_sim(
        cluster: data_processor.Cluster,
    k: int = 10,
):
    """
    For now only supports single vector query
    """
    # TODO:
    # 1. data persistence
    # 2. calculate qps
    base, query, ground_truth = get_dataset(dataset)
    queryLen = query.shape[0]
    print(base.shape)
    num, dim = base.shape
    random_index = np.random.randint(0, queryLen)

    oneQuery = query[random_index][None, :]
    node_dict: dict = cluster.get_nodes_dict(oneQuery, k)
    final_res = []
    final_dist = []
    maximum_search_time = 0
    search_times = []
    for node, bucket_list in node_dict.items():
        node_start = time.time()
        res, dist = node.query(oneQuery, bucket_list, k)
        final_res.extend(res.tolist())
        final_dist.extend(dist.tolist())
        current_search_time = time.time() - node_start
        search_times.append(current_search_time)
        if current_search_time > maximum_search_time:
            maximum_search_time = current_search_time
    final_res = np.array(final_res)
    final_dist = np.array(final_dist)
    sort_indices = np.argsort(final_dist, axis=0)
    top_k_vectors = final_res[sort_indices[:k]]
    top_k_distances = final_dist[sort_indices[:k]]
    # recalculate distances
    recalculated_distances = np.linalg.norm(top_k_vectors - oneQuery, axis=1)
    ground_truth_labels = ground_truth[random_index][:k]
    ground_truth_vectors = base[ground_truth_labels]
    ground_truth_distances = np.linalg.norm(ground_truth_vectors - oneQuery, axis=1)
    print(
        "recall: ",
        utils.calculate_recall_by_vector(ground_truth_vectors, top_k_vectors),
    )
    print("average search time: ", np.mean(search_times))
    print("maximum search time: ", maximum_search_time)


if __name__ == "__main__":
    num_buckets = 200
    num_nodes = 10
    dataset = "sift"
    base, query, ground_truth = get_dataset(dataset)
    # create_persistent_cluster(base, num_buckets, num_nodes, "kmeans", dataset)
    cluster = load_persistent_cluster(dataset)

    queryLen = query.shape[0]
    print(base.shape)
    num, dim = base.shape
    distributed_search_sim(cluster, k=22)

