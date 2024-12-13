from typing import Literal, Union
from hnsw import HNSW
import numpy as np
from sklearn.cluster import KMeans


class Node:
    """
    Each node contains:
    buckets: list of hnsw
    data_list: list of data, corresponds to the hnsw
    """

    def __init__(
        self,
        buckets: list,
        ef_construction: int,
        thread_num: int,
        M: int,
        offset: int,
    ):
        self.offset = offset
        self.centroids = [bucket[1] for bucket in buckets]
        data_list = [np.array(bucket[0]) for bucket in buckets]
        dim = data_list[0].shape[1]
        self.buckets = []
        for i in range(len(buckets)):
            hnsw = HNSW(
                dim=dim,
                max_elements=data_list[i].shape[0],
                ef_construction=ef_construction,
                thread_num=thread_num,
                M=M,
                index="index" + str(i) + ".bin",
                nth=i,
            )
            hnsw.add_items(data_list[i])
            self.buckets.append(hnsw)
        self.data_list = [np.array(data) for data in data_list]

    def get_bucket_count(self):
        return len(self.buckets)

    def get_centroids(self, buckets, with_offset: bool = True):
        if buckets is None:
            # return all centroids
            return self.centroids
        if with_offset:
            buckets = [bucket - self.offset for bucket in buckets]
        return [self.centroids[i] for i in buckets]

    def query(
        self, vector: np.ndarray, buckets: list, k: int, with_offset: bool = True
    ):
        """
        Query the node
        :param vector: query vector
        :param buckets: list of hnsw
        :param k: number of nearest neighbors to return
        :return: relative vectors and distances
        """
        result_vectors_lst = []
        result_distances_lst = []
        if with_offset:
            buckets = [bucket - self.offset for bucket in buckets]
        for i in buckets:
            hnsw = self.buckets[i]
            if hnsw.index.get_current_count() <= k:
                labels, distances = hnsw.query(vector, hnsw.index.get_current_count())
            else:
                labels, distances = hnsw.query(vector, k)

            # TODO: apply multi vector query
            labels = labels[0]
            distances = distances[0]
            vecs = np.array(self.data_list[i])
            vectors = vecs[labels].tolist()
            result_vectors_lst.extend(vectors)
            result_distances_lst.extend(distances)
        result_vectors_lst = np.array(result_vectors_lst)
        result_distances_lst = np.array(result_distances_lst)
        sorted_index = np.argsort(result_distances_lst, axis=0)
        result_vectors_lst = result_vectors_lst[sorted_index[:k]]
        result_distances_lst = result_distances_lst[sorted_index[:k]]

        return result_vectors_lst, result_distances_lst

    def bf_query(self, query: np.ndarray, k: int, return_all: bool = False):
        """
        Query all buckets, return top k vectors and distances and bucket index
        """
        result_vectors_lst = []
        result_distances_lst = []
        bucket_indices = []
        for i, hnsw in enumerate(self.buckets):
            currnt_element_count = hnsw.index.get_current_count()
            actual_k = min(k, currnt_element_count)
            labels, distances = hnsw.query(query, actual_k)
            labels = labels[0]
            distances = distances[0]
            vecs = np.array(self.data_list[i])
            vectors = vecs[labels].tolist()
            result_vectors_lst.extend(vectors)
            result_distances_lst.extend(distances)
            bucket_indices.extend([i + self.offset] * actual_k)
        result_vectors_lst = np.array(result_vectors_lst)
        result_distances_lst = np.array(result_distances_lst)
        bucket_indices = np.array(bucket_indices)
        sort_indices = np.argsort(result_distances_lst, axis=0)
        top_k_vectors = result_vectors_lst[sort_indices]
        top_k_distances = result_distances_lst[sort_indices]
        top_k_bucket_indices = bucket_indices[sort_indices]
        if not return_all:
            top_k_vectors = top_k_vectors[:k]
            top_k_distances = top_k_distances[:k]
            top_k_bucket_indices = top_k_bucket_indices[:k]
        return (
            top_k_vectors.tolist(),
            top_k_distances.tolist(),
            top_k_bucket_indices.tolist(),
        )

    def get_bucket_lens(self):
        return [hnsw.index.get_current_count() for hnsw in self.buckets]


class Cluster:
    def __init__(
        self,
        nodes: list,
        bucket_dict: dict,
        centroids: np.ndarray,
        ef_construction: int,
        thread_num: int,
        M: int,
        raw_data: np.ndarray,
    ):
        self.nodes = nodes
        self.bucket_dict = bucket_dict
        self.centroids = centroids
        dim = centroids.shape[1]
        self.hnsw = HNSW(
            dim=dim,
            max_elements=centroids.shape[0],  # TODO: need to be modifiable
            ef_construction=ef_construction,
            thread_num=thread_num,
            M=M,
            index="cluster.bin",
        )
        self.hnsw.add_items(centroids)
        self.raw_data = raw_data
        self.raw_hnsw = HNSW(
            dim=raw_data.shape[1],
            max_elements=raw_data.shape[0],
            ef_construction=ef_construction,
            thread_num=thread_num,
            M=M,
            index="raw.bin",
        )
        self.raw_hnsw.add_items(raw_data)

    def get_nodes_dict(self, query: np.ndarray, top_k: int):
        """
        Find the nodes whose buckets' centroids are closest to the query
        returns: { node1: [bucket_index1, bucket_index2, ...], node2: [bucket_index1, bucket_index2, ...], ... }
        Index with offset
        """
        labels, dists = self.hnsw.query(query, top_k)
        print("k:", top_k)
        nodes_dict = {}
        for label in labels[0].tolist():
            node = self.bucket_dict[label]
            if node not in nodes_dict:
                nodes_dict[node] = []
            nodes_dict[node].append(label)
        return nodes_dict

    def get_buckets(self, labels:list):
        """
        Get the buckets from labels
        """
        return [self.bucket_dict[label] for label in labels]

    def bf_query(self, query: np.ndarray, k: int, return_all: bool = False):
        """
        Returns:
        top k vectors and distances
        nodes and buckets in each node
        """
        node_indices = []
        dists = []
        meta_bucket_indices = []
        vectors = []
        for i in range(len(self.nodes)):
            res, dist, bucket_indices = self.nodes[i].bf_query(
                query, k, return_all=True
            )
            node_indices.extend([i] * len(res))
            vectors.extend(res)
            dists.extend(dist)
            meta_bucket_indices.extend(bucket_indices)
        vectors = np.array(vectors)
        dists = np.array(dists)
        node_indices = np.array(node_indices)
        meta_bucket_indices = np.array(meta_bucket_indices)
        sort_indices = np.argsort(dists, axis=0)
        dists = dists[sort_indices]
        vectors = vectors[sort_indices]
        node_indices = node_indices[sort_indices]
        meta_bucket_indices = meta_bucket_indices[sort_indices]
        if not return_all:
            vectors = vectors[:k]
            dists = dists[:k]
            node_indices = node_indices[:k]
            meta_bucket_indices = meta_bucket_indices[:k]
        dists = np.sqrt(dists)
        return vectors, dists, node_indices, meta_bucket_indices

    def get_total_bucket_count(self):
        return sum([node.get_bucket_count() for node in self.nodes])

    def get_nodes_count(self):
        return len(self.nodes)

    def show_nodes_info(self):

        bucket_lens = []
        for node in self.nodes:
            bucket_lens.extend(node.get_bucket_lens())
        print(f"Minimum hnsw element count: {min(bucket_lens)}")
        print(f"Maximum hnsw element count: {max(bucket_lens)}")
        print(f"Average hnsw element count: {np.mean(bucket_lens)}")
        print(
            "Bucket count of each node: ",
            [node.get_bucket_count() for node in self.nodes],
        )
        print("Total number of elements:")
        tatal_elements = 0
        for node in self.nodes:
            tatal_elements += sum(
                [hnsw.index.get_current_count() for hnsw in node.buckets]
            )
        print(tatal_elements)

    def get_bucket_centroids(self, labels):
        """
        Get the centroids of the buckets
        """
        return self.centroids[labels]

    def get_all_buckets(self):
        """
        Get all buckets
        """
        return self.bucket_dict.values()


EF = 200
M = 32
K = 10
PARTITION_NUM = 10
THREAD_NUM = 16

class BucketCluster:
    # TODO: finish

    def __init__(self, data: np.ndarray, num_clusters: int, index_type: Literal["hnsw", "bf"]):
        self.data = data
        self.num_clusters = num_clusters
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
        self.labels = kmeans.fit_predict(data)
        self.centroids = kmeans.cluster_centers_
        if index_type == "bf":
            self.index = HNSW(
                    dim=data.shape[1],
                    max_elements=data.shape[0],
                    ef_construction=EF,
                    thread_num=THREAD_NUM,
                    M=M,
                    index="cluster.bin",
                    index_type="bf",
                    )

    def retrieve_data_with_centroids(self, q: np.ndarray, k: int):
        """
        Retrieve data with centroids
        """
        labels = np.argsort(np.linalg.norm(self.centroids - q, axis=1))[:k]
        indices = np.where(np.isin(self.labels, labels))[0]
        return self.data[indices], indices


