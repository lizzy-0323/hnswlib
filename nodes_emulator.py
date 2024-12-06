from typing import Literal
import time
from joblib import dump
from matplotlib import pyplot
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
import node
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


def load_persistent_cluster(dataset_name) -> node.Cluster:
    """
    Load a persistent cluster
    """
    with open(f"./data/pickle/{dataset_name}_cluster.pkl", "rb") as f:
        cluster = pickle.load(f)
    return cluster


def distributed_search_sim(
        cluster: node.Cluster,
    k: int = 10,
):
    """
    For now only supports single vector query
    """
    # TODO:
    # 1. calculate qps
    print(f"Total number of buckets in cluster: {cluster.get_total_bucket_count()}")
    print(f"Total number of nodes in cluster: {cluster.get_nodes_count()}")
    base, query, ground_truth = get_dataset(dataset)
    queryLen = query.shape[0]
    print(base.shape)
    random_index = np.random.randint(0, queryLen)
    oneQuery = query[random_index][None, :]
    node_dict: dict = cluster.get_nodes_dict(oneQuery, k)
    cenoids = []
    for node, bucket_list in node_dict.items():
        cenoids.extend(node.get_centroids(bucket_list))
    cenoids = np.array(cenoids)
    final_res = []
    final_dist = []
    maximum_search_time = 0
    search_times = []
    print("Number of nodes for query: ", len(node_dict))
    print("Number of buckets for query: ", sum([len(bucket_list) for bucket_list in node_dict.values()]))
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
    bf_top_k_vectors, bf_top_k_distances, node_indices, bucket_indices = cluster.bf_query(oneQuery, k)
    # recalculate distances
    ground_truth_labels = ground_truth[random_index][:k]
    ground_truth_vectors = base[ground_truth_labels]
    ground_truth_distances = np.linalg.norm(ground_truth_vectors - oneQuery, axis=1)
    centroid_distances = np.linalg.norm(cenoids - oneQuery, axis=1)
    recall = utils.calculate_recall_by_vector(ground_truth_vectors, top_k_vectors)
    print(
        "recall: ",
        recall,
    )
    bf_recall = utils.calculate_recall_by_vector(ground_truth_vectors, bf_top_k_vectors)
    print(
        "bf recall: ",
        bf_recall,
    )
    bf_bucket_unique_indices = set(np.unique(bucket_indices).tolist())
    bf_node_unique_indices = set(np.unique(node_indices).tolist())
    _distributed_bucket_indices = []
    _distributed_node_indices = []
    for node, bucket_list in node_dict.items():
        _distributed_bucket_indices.extend(bucket_list)
        node_index = cluster.nodes.index(node)
        _distributed_node_indices.append(node_index)
    distributed_bucket_unique_indices = set(_distributed_bucket_indices)
    distributed_node_unique_indices = set(_distributed_node_indices)
    # print bucket indices and node indices, sorted for better comparison
    print("bf bucket unique indices: ", sorted(list(bf_bucket_unique_indices)))
    print("distributed bucket unique indices: ", sorted(list(distributed_bucket_unique_indices)))
    bucket_recall = len(bf_bucket_unique_indices & distributed_bucket_unique_indices) / len(bf_bucket_unique_indices)
    node_recall = len(bf_node_unique_indices & distributed_node_unique_indices) / len(bf_node_unique_indices)
    print(f"bucket recall: {bucket_recall}")
    print(f"node recall: {node_recall}")

    print("bf top k distances: ", bf_top_k_distances)
    print("ground truth distances: ", ground_truth_distances)
    print("distributed distances: ", np.sqrt(top_k_distances))
    print("distributed centroid distances: ", centroid_distances)

    bf_bucket_centroids = cluster.get_bucket_centroids(np.array(list(bf_bucket_unique_indices)))
    bf_bucket_centroid_distances = np.linalg.norm(bf_bucket_centroids - oneQuery, axis=1)
    print("bf bucket centroid distances: ", bf_bucket_centroid_distances)


    print("average search time: ", np.mean(search_times))
    print("maximum search time: ", maximum_search_time)
    return recall, bucket_recall, node_recall


if __name__ == "__main__":
    num_buckets = 200
    num_nodes = 20
    repeat_num = 5
    dataset = "sift"
    base, query, ground_truth = get_dataset(dataset)
    # create_persistent_cluster(base, num_buckets, num_nodes, "kmeans", dataset)
    cluster = load_persistent_cluster(dataset)
    cluster.show_nodes_info()
    k_x = []
    recall_y = []
    node_recall_y = []
    bucket_recall_y = []

    for i in range(2, 23):
        average_recall = 0
        average_bucket_recall = 0
        average_node_recall = 0
        for j in range(repeat_num):
            single_epoch_recall, single_epoch_bucket_recall, single_epoch_node_recall = distributed_search_sim(cluster, k=i)
            average_recall += single_epoch_recall / repeat_num
            average_bucket_recall += single_epoch_bucket_recall / repeat_num
            average_node_recall += single_epoch_node_recall / repeat_num
        k_x.append(i)
        recall_y.append(average_recall)
        node_recall_y.append(average_node_recall)
        bucket_recall_y.append(average_bucket_recall)

    print(k_x)
    print(recall_y)
    # plot three curves in one image
    pyplot.xlabel("k")
    pyplot.ylabel("recall")
    pyplot.plot(k_x, recall_y)
    pyplot.plot(k_x, node_recall_y)
    pyplot.plot(k_x, bucket_recall_y)
    pyplot.legend(["recall", "node recall", "bucket recall"])
    pyplot.savefig(f"./data/image/{dataset}_recall_vs_k.png")


