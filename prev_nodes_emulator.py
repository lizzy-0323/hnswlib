
from typing import Dict, List, Literal, Union

from tqdm import tqdm
from hnsw import HNSW
import numpy as np
from sklearn.cluster import KMeans


EF = 200
M = 32
K = 10
PARTITION_NUM = 10
THREAD_NUM = 16





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


class Bucket2:
    '''
    does NOT store:
        - centroids
    '''
    absolute_labels: np.ndarray            # stores absolute global labels
    hnsw_index: HNSW
    bf_index: HNSW
    data: np.ndarray
    def __init__(
            self,
            hnsw_index: HNSW,
            bf_index: HNSW,
            data: np.ndarray,
            absolute_labels: np.ndarray,
            ):
        self.hnsw_index = hnsw_index
        self.bf_index = bf_index
        self.data = data
        self.absolute_labels = absolute_labels

    def query(self,
              query: np.ndarray,
              k: int,
              index_type: Literal["hnsw", "bf"] = "hnsw",
              relative: bool = False,
              ):
        '''
        :param relative: if True, returns relative labels inside this bucket

        returns: (labels, l2distances), one dimensional, as it does not support multi-vector search
        '''
        if index_type == "hnsw":
            _relative_labels, _l2distances = self.hnsw_index.query(query, k)
        else:
            _relative_labels, _l2distances = self.bf_index.query(query, k)
        if relative:
            return _relative_labels[0], _l2distances[0]
        else:
            return self.absolute_labels[_relative_labels][0], _l2distances[0]

    def query_vectors(
            self, query: np.ndarray, k: int, index_type: Literal["hnsw", "bf"] = "hnsw",
            ):
        if index_type == "hnsw":
            _relative_labels, _l2distances = self.hnsw_index.query(query, k)
        else:
            _relative_labels, _l2distances = self.bf_index.query(query, k)
        vecs = self.data[_relative_labels]
        return vecs, _l2distances

    @classmethod
    def from_data(
            cls,
            data: np.ndarray,
            label: np.ndarray,
            ef_construction: int,
            thread_num: int,
            M: int,
            ):
        hnsw = HNSW(
            dim=data.shape[1],
            max_elements=data.shape[0],
            ef_construction=ef_construction,
            thread_num=thread_num,
            M=M,
            index="hnsw_index.bin",
            index_type="hnsw",
            )
        bf = HNSW(
            dim=data.shape[1],
            max_elements=data.shape[0],
            ef_construction=ef_construction,
            thread_num=thread_num,
            M=M,
            index="bf_index.bin",
            index_type="bf",
            )
        hnsw.add_items(data)
        bf.add_items(data)
        return cls(hnsw_index=hnsw, bf_index=bf, data=data, absolute_labels=label)

class Node2:
    buckets: List[Bucket2]


    centroids: np.ndarray                   # duplicate of part of centroids in Cluster
    bucket_dict: Dict[int, Bucket2]

    def __init__(
            self,
            buckets: List[Bucket2],
            bucket_indices: np.ndarray,
            centroids: np.ndarray,
            ):
        self.buckets = buckets
        self.centroids = centroids
        self.bucket_dict = {i: bucket for i, bucket in zip(bucket_indices, buckets)}

    @property
    def num_buckets(self):
        return len(self.buckets)

    def _query_labels(
            self,
            vector: np.ndarray,
            k: int,
            bucket_indices: Union[None, np.ndarray] = None,
            index_type: Literal["hnsw", "bf"] = "hnsw",
            sqrt: bool = True,
            relative: bool = False,
            ):
        _bucket_indices:list = list(self.bucket_dict.keys()) if bucket_indices is None else bucket_indices.tolist()
        result = np.empty((0)).astype(np.int32)
        dist = np.empty((0))
        bucket_indices = np.empty((0)).astype(np.int32)
        for _bucket_index in _bucket_indices:
            _bucket = self.bucket_dict[_bucket_index]
            _labels, _dist = _bucket.query(vector, k, index_type=index_type, relative=relative)
            result = np.concatenate((result, _labels))
            dist = np.concatenate((dist, _dist))
            bucket_indices = np.concatenate((bucket_indices, np.ones_like(_labels) * _bucket_index))
        sort_indices = np.argsort(dist)
        if sqrt:
            return result[sort_indices], bucket_indices, np.sqrt(dist[sort_indices])
        else:
            return result[sort_indices], bucket_indices, dist[sort_indices]


    def query_abs_labels(
            self,
            vector: np.ndarray,
            k: int,
            bucket_indices: Union[None, np.ndarray] = None,
            index_type: Literal["hnsw", "bf"] = "hnsw",
            sqrt: bool = True,
            ):
        '''
        sqrt: if True, returns sqrt of distances
        returns: absolute labels, distances, sorted
        '''
        abs_indices, _, dist = self._query_labels(
            vector,
            k,
            bucket_indices,
            index_type,
            sqrt,
            relative=False,
            )
        return abs_indices, dist

    def query_relative_labels(
            self,
            vector: np.ndarray,
            k: int,
            bucket_indices: Union[None, np.ndarray] = None,
            index_type: Literal["hnsw", "bf"] = "hnsw",
            sqrt: bool = True,
            ):
        return self._query_labels(
            vector,
            k,
            bucket_indices,
            index_type,
            sqrt,
            relative=False,
            )



    def query_vectors(
            self,
            vector: np.ndarray,
            k: int,
            bucket_indices: Union[None, np.ndarray] = None,
            index_type: Literal["hnsw", "bf"] = "hnsw",
            sqrt: bool = True,
            ):
        '''
        currently only supports synchronous query
        '''
        # TODO: implement distributed search
        _bucket_indices:list = list(self.bucket_dict.keys()) if bucket_indices is None else bucket_indices.tolist()
        result_vectors = np.empty((0, vector.shape[0]))
        dist = np.empty((0))
        bucket_indices = np.empty((0)).astype(np.int32)
        for _bucket_index in _bucket_indices:
            _bucket = self.bucket_dict[_bucket_index]
            _vectors, _dist = _bucket.query_vectors(vector, k, index_type=index_type)
            result_vectors = np.concatenate((result_vectors, _vectors), axis=0)
            dist = np.concatenate((dist, _dist))
        sort_indices = np.argsort(dist)
        return result_vectors[sort_indices], dist[sort_indices] if not sqrt else np.sqrt(dist[sort_indices])



class Cluster2:
    _labels: np.ndarray                 # 0, 1, 2, 3, ..., n        for generating stuff
    data: np.ndarray
    _num_buckets: int
    buckets_labels: np.ndarray          # shape: (data.shape[0],) range: [0, num_buckets)
    bucket_centroids: np.ndarray
    node_labels: np.ndarray             # shape: (num_buckets,) range: [0, num_nodes)
    nodes: List[Node2]
    top_hnsw_index: HNSW
    top_bf_index: HNSW
    _bucket_labels: np.ndarray          # 0, 1, 2, 3, ..., num_buckets-1

    def __init__(
            self,
            data: np.ndarray,
            bucket_labels: np.ndarray,
            bucket_centroids: np.ndarray,
            nodes_labels: np.ndarray,
            nodes: List[Node2],
            top_hnsw_index: HNSW,
            top_bf_index: HNSW,
            ):
        self.data = data
        self._labels = np.arange(data.shape[0])
        self.buckets_labels = bucket_labels
        self.node_labels = nodes_labels
        self.bucket_centroids = bucket_centroids
        self.nodes = nodes
        self._num_buckets = bucket_centroids.shape[0]
        self.top_hnsw_index = top_hnsw_index
        self.top_bf_index = top_bf_index
        self._bucket_labels = np.arange(self._num_buckets)

    @classmethod
    def FromPrePartitionedBuckets(
            cls,
            data: np.ndarray,
            bucket_labels: np.ndarray,
            bucket_centroids: np.ndarray,
            num_nodes: int,
            bucket_cluster_method: Literal["kmeans", "spectral"] = "kmeans",
            bucket_ef_construction: int=EF,
            bucket_thread_num: int=THREAD_NUM,
            bucket_M: int=M,
            bucket_index: str="node_index.bin",
            node_ef_construction: int=EF,
            node_thread_num: int=THREAD_NUM,
            node_M: int=M,
            node_index: str="node_index.bin",
            ):

        num_buckets = bucket_centroids.shape[0]
        _labels = np.arange(data.shape[0])

        buckets_node_labels = np.zeros(num_buckets, dtype=np.int32)
        if bucket_cluster_method == "kmeans":
            bucket_kmeans = KMeans(n_clusters=num_nodes)
            buckets_node_labels = bucket_kmeans.fit_predict(bucket_centroids)
        else:
            # TODO: implement spectral clustering
            raise ValueError(f"Unsupported bucket cluster method: {bucket_cluster_method}")
        # create nodes
        nodes:List[Node2] = []
        for i in tqdm(range(num_nodes)):
            current_node_bucket_indices = np.where(buckets_node_labels == i)[0]
            current_node_bucket_indices_list = current_node_bucket_indices.tolist()
            node_buckets = []
            for _bucket_index in current_node_bucket_indices_list:
                current_bucket_data = data[bucket_labels == _bucket_index]
                current_bucket_indices = _labels[bucket_labels == _bucket_index]
                bucket = Bucket2.from_data(
                    current_bucket_data,
                    current_bucket_indices,
                    bucket_ef_construction,
                    bucket_thread_num,
                    bucket_M,
                    )
                node_buckets.append(bucket)
            _current_node_bucket_centroids = bucket_centroids[current_node_bucket_indices_list]
            nodes.append(Node2(
                buckets=node_buckets,
                bucket_indices=current_node_bucket_indices,
                centroids=_current_node_bucket_centroids
                ))
        top_hnsw_index = HNSW(
            dim=bucket_centroids.shape[1],
            max_elements=bucket_centroids.shape[0],
            ef_construction=node_ef_construction,
            thread_num=node_thread_num,
            M=node_M,
            index=node_index,
            index_type="hnsw",
            )
        top_bf_index = HNSW(
            dim=bucket_centroids.shape[1],
            max_elements=bucket_centroids.shape[0],
            ef_construction=node_ef_construction,
            thread_num=node_thread_num,
            M=node_M,
            index=node_index,
            index_type="hnsw",
            )
        top_hnsw_index.add_items(bucket_centroids)
        top_bf_index.add_items(bucket_centroids)
        return cls(
                data=data,
                bucket_labels=bucket_labels,
                bucket_centroids=bucket_centroids,
                nodes_labels=buckets_node_labels,
                nodes=nodes,
                top_hnsw_index=top_hnsw_index,
                top_bf_index=top_bf_index,
                )


    @classmethod
    def FromData(
            cls,
            data: np.ndarray,
            num_buckets: int,
            num_nodes: int,
            bucket_ef_construction: int=EF,
            bucket_thread_num: int=THREAD_NUM,
            bucket_M: int=M,
            bucket_index: str="node_index.bin",
            node_ef_construction: int=EF,
            node_thread_num: int=THREAD_NUM,
            node_M: int=M,
            node_index: str="node_index.bin",
            vec_cluster_method: Literal["kmeans"] = "kmeans",       # for now only kmeans is supported
            bucket_cluster_method: Literal[
                "kmeans",
                "spectral",
                # "graph",      # graph isn't supported for now
                ] = "kmeans",

            ):
        # cluster data
        if vec_cluster_method == "kmeans":
            vec_kmeans = KMeans(n_clusters=num_buckets)
            bucket_labels = vec_kmeans.fit_predict(data)
            bucket_centroids = vec_kmeans.cluster_centers_
        else:
            raise ValueError(f"Unsupported vector cluster method: {vec_cluster_method}")
        # cluster buckets
        return cls.FromPrePartitionedBuckets(
            data,
            bucket_labels,
            bucket_centroids,
            num_nodes,
            bucket_cluster_method,
            bucket_ef_construction,
            bucket_thread_num,
            bucket_M,
            bucket_index,
            node_ef_construction,
            node_thread_num,
            node_M,
            node_index,
            )

    @property
    def num_buckets(self):
        return self._num_buckets

    @property
    def num_nodes(self):
        return len(self.nodes)

    def get_top_k_buckets(
            self,
            query: np.ndarray,
            k: int,
            index_type: Literal["hnsw", "bf"] = "hnsw",
            ):
        '''
        returns a dict: {node_index: [bucket_indices: np.ndarray]}
        '''
        _labels, _ = cluster.top_hnsw_index.query(randomquery, k)
        _labels = _labels[0]
        result = {}
        bucket_nodes = cluster.node_labels[_labels]
        _nodes = np.unique(bucket_nodes)

        for _node in _nodes:
            _node_mask = bucket_nodes == _node
            result[_node] = _labels[_node_mask]
        return result

    def query(
            self,
            query: np.ndarray,
            top_k_vectors: int,
            top_k_buckets: int,
            index_type: Literal["hnsw", "bf"] = "hnsw",
            ):
        buckets_dict = self.get_top_k_buckets(query, top_k_buckets, index_type=index_type)
        labels = np.empty((0))
        dists = np.empty((0))
        for _node, _buckets in buckets_dict.items():
            _node_labels, _node_dists = self.nodes[_node].query_abs_labels(
                query,
                top_k_vectors,
                _buckets,
                index_type=index_type,
                sqrt=False,
                    )
            labels = np.concatenate((labels, _node_labels))
            dists = np.concatenate((dists, _node_dists))
        sort_indices = np.argsort(dists)
        return labels[sort_indices[top_k_vectors]], dists[sort_indices[top_k_vectors]]





