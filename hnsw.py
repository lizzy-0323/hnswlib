import hnswlib
import numpy as np


class HNSW:
    def __init__(
        self,
        dim: int,
        max_elements: int,
        ef_construction: int,
        thread_num: int,
        M: int,
        index: str,
        nth: int = 0,
        index_type: str = "hnsw",
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
            self.index.set_ef(20)
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
