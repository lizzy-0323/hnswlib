from typing import Literal, Union
import time
from matplotlib import pyplot
import data_processor
import numpy as np
from hnsw import HNSW
from data_loader import get_dataset
import node
from utils import Timer
from node import Node
import utils
import pickle
from sklearn.cluster import KMeans

timer = Timer()

# TODO:
# [ ] calculate node recall rate
# [ ] calculate bucket recall rate
# [ ] alter func return values to support better analysis

def create_persistent_kmeans_data(data, num_clusters, num_buckets, path):
    """
    Create a persistent kmean object
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    data_dict = {
        "base": data,
        "labels": labels,
        "centroids": centroids,
    }
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)
    return data_dict


def load_presistent_kmeans_data(path):
    with open (path, "rb") as f:
        saved_dict = pickle.load(f)
    return saved_dict

def distributed_search_sim(
        cluster: node.Cluster,
        query: np.ndarray,
    k: int = 10,
    buckets_k: int = 10,
    ground_truth: Union[np.ndarray, None] = None,
    index_type: Literal["bf", "hnsw"] = "hnsw",
    ):
    """
    For now only supports single vector query
    """
    # TODO:
    # 1. calculate qps
    data = cluster.data
    print(f"Total number of buckets in cluster: {cluster.num_buckets}")
    print(f"Total number of nodes in cluster: {cluster.num_nodes}")
    queryLen = query.shape[0]
    print(base.shape)
    random_index = np.random.randint(0, queryLen)
    random_query = query[random_index]
    maximum_search_time = 0
    search_times = []
    node_dict = cluster.get_node_dict(random_query, buckets_k)
    print("Number of nodes for query: ", len(node_dict))
    print("Number of buckets for query: ", sum([len(bucket_list) for bucket_list in node_dict.values()]))
    query_res_labels = np.empty((0), dtype=np.int32)
    query_res_dists = np.empty((0), dtype=np.float32)
    for node_index, bucket_list in node_dict.items():
        node_start = time.time()
        _node_labels, _node_dists = cluster.nodes[node_index].query_abs_labels(
                random_query,
                k,
                bucket_list,
                index_type=index_type
                )
        _node_labels = _node_labels
        _node_dists = _node_dists
        query_res_labels = np.concatenate((query_res_labels, _node_labels))
        query_res_dists = np.concatenate((query_res_dists, _node_dists))
        current_search_time = time.time() - node_start
        search_times.append(current_search_time)
        if current_search_time > maximum_search_time:
            maximum_search_time = current_search_time
    sort_indices = np.argsort(query_res_dists, axis=0)
    top_k_labels = query_res_labels[sort_indices[:k]]
    top_k_distances = query_res_dists[sort_indices[:k]]
    if ground_truth is None:
        print("No ground truth provided, using brute force search as ground truth")
        bf_top_k_labels, bf_top_k_distances = cluster.query(random_query, k, cluster.num_buckets, index_type="bf")
        ground_truth_labels = bf_top_k_labels
        ground_truth_distances = bf_top_k_distances
        ground_truth_vectors = base[bf_top_k_labels]
    else:
        ground_truth_labels = ground_truth[random_index][:k]
        ground_truth_vectors = base[ground_truth_labels]
        ground_truth_distances = np.linalg.norm(ground_truth_vectors - random_query, axis=1)
    # print(ground_truth_labels, top_k_labels)
    ground_truth_set = set(ground_truth_labels.tolist())
    top_k_set = set(top_k_labels.tolist())
    recall = len(ground_truth_set & top_k_set) / k
    # print(ground_truth_distances)
    # print(top_k_distances)
    print(
        "recall: ",
        recall,
    )
    print("average search time: ", np.mean(search_times))
    print("maximum search time: ", maximum_search_time)
    print("qps: ", k/maximum_search_time)


num_buckets = 200
buckets_k = 20
num_nodes = 20
repeat_num = 1
dataset = "sift"
query_type = "hnsw"             # set to bf to use brute force search, hnsw to use hnsw search
base, query, ground_truth = get_dataset(dataset)
# data_dict = create_persistent_kmeans_data(base, num_clusters=num_buckets, num_buckets=num_buckets, path=f"./data/pickle/{dataset}_kmeans_data.pkl")
data_dict = load_presistent_kmeans_data(f"./data/pickle/{dataset}_kmeans_data.pkl")
cluster = node.Cluster.FromPrePartitionedBuckets(
        data=base,
        bucket_labels=data_dict["labels"],
        bucket_centroids=data_dict["centroids"],
        num_nodes=num_nodes,
        )
k_x = []
recall_y = []
node_recall_y = []
bucket_recall_y = []

for i in range(90, 96):
    average_recall = 0
    average_bucket_recall = 0
    average_node_recall = 0
    for j in range(repeat_num):
        print(f"Round {j} under k={i}")
        distributed_search_sim(cluster, k=i, buckets_k=buckets_k, query=query, ground_truth=ground_truth, index_type=query_type)
    k_x.append(i)
    recall_y.append(average_recall)
    node_recall_y.append(average_node_recall)
    bucket_recall_y.append(average_bucket_recall)

# print(k_x)
# print(recall_y)
# pyplot.xlabel("k")
# pyplot.ylabel("recall")
# pyplot.plot(k_x, recall_y)
# pyplot.plot(k_x, node_recall_y)
# pyplot.plot(k_x, bucket_recall_y)
# pyplot.legend(["recall", "node recall", "bucket recall"])
# pyplot.savefig(f"./data/image/{dataset}_recall_vs_k.png")



