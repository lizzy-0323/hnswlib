from typing import Literal
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

timer = Timer()


def distributed_search_sim(
    num_buckets: int = 200,
    num_nodes: int = 10,
    partition_method: Literal["kmeans", "spectral"] = "kmeans",
    k: int = 10,
):
    """
    For now only supports single vector query
    """
    base, query, ground_truth = get_dataset("sift")
    queryLen = query.shape[0]
    print(base.shape)
    num, dim = base.shape
    cluster = build_cluster(base, num_buckets, num_nodes, partition_method)
    oneQuery = query[0][None, :]
    node_dict: dict = cluster.get_nodes_dict(oneQuery, k)
    print(node_dict)
    final_res = []
    final_dist = []
    for node, bucket_list in node_dict.items():
        print(oneQuery, bucket_list)
        res, dist = node.query(oneQuery, bucket_list, k)
        final_res.extend(res.tolist())
        final_dist.extend(dist.tolist())
    final_res = np.array(final_res)
    final_dist = np.array(final_dist)
    sort_indices = np.argsort(final_dist, axis=0)
    top_k_vectors = final_res[sort_indices[:k]]
    top_k_distances = final_dist[sort_indices[:k]]
    print("top_k_vectors: ", top_k_vectors)
    print("top_k_distances: ", top_k_distances)
    # recalculate distances
    recalculated_distances = np.linalg.norm(top_k_vectors - oneQuery, axis=1)
    print("recalculated_distances: ", recalculated_distances)
    

    ground_truth_labels = ground_truth[0][:k]
    ground_truth_vectors = base[ground_truth_labels]
    ground_truth_distances = np.linalg.norm(ground_truth_vectors - oneQuery, axis=1)
    print("ground_truth_vectors: ", ground_truth_vectors)
    print("ground_truth_distances: ", ground_truth_distances)
    print("recall: ", utils.calculate_recall_by_vector(ground_truth_vectors, top_k_vectors))



if __name__ == "__main__":
    distributed_search_sim(k = 22)
