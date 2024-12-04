from hnsw import HNSW
import numpy as np


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
        for i, hnsw in enumerate(self.buckets):
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
        # sort by result_distances_lst, return top k
        sort_indices = np.argsort(result_distances_lst, axis=0)
        top_k_vectors = result_vectors_lst[sort_indices[:k]]
        top_k_distances = result_distances_lst[sort_indices[:k]]

        return top_k_vectors, top_k_distances


class Cluster:
    def __init__(
        self,
        nodes: list,
        bucket_dict: dict,
        centroids: np.ndarray,
        ef_construction: int,
        thread_num: int,
        M: int,
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


    def get_nodes_dict(self, query:np.ndarray, top_k:int):
        """
        Find the nodes whose buckets' centroids are closest to the query
        returns: { node1: [bucket_index1, bucket_index2, ...], node2: [bucket_index1, bucket_index2, ...], ... }
        Index with offset
        """
        labels, dists = self.hnsw.query(query, top_k)
        nodes_dict = {}
        print("Distances:", dists)
        for label in labels[0].tolist():
            node = self.bucket_dict[label]
            if node not in nodes_dict:
                nodes_dict[node] = []
            nodes_dict[node].append(label)
        return nodes_dict


