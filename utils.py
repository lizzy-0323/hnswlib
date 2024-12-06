import numpy as np
import time


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


def calculate_recall(ground_truth: np.ndarray, labels: np.ndarray, top_k: int) -> float:
    recall = 0
    assert len(ground_truth) == len(
        labels
    ), "The number of ground truth and labels should be the same."
    for real_candidate, candidate in zip(ground_truth, labels):
        recall += len(set(real_candidate) & set(candidate[:top_k])) / top_k
    return recall / len(ground_truth)



def calculate_recall_by_vector(ground_truth_vectors: np.ndarray, retrieved_sorted_vectors: np.ndarray) -> float:
    """
    Calculate recall by comparing ground truth and retrieved sorted vector
    """
    recall = 0
    assert len(ground_truth_vectors) == len(
        retrieved_sorted_vectors
    ), "The number of ground truth and retrieved vector should be the same."
    _ground_truth_list = ground_truth_vectors.tolist()
    _retrieved_sorted_list = retrieved_sorted_vectors.tolist()
    ground_truth_set = set(tuple(vec) for vec in _ground_truth_list)
    retrieved_sorted_set = set([tuple(vec) for vec in _retrieved_sorted_list])
    return len(ground_truth_set & retrieved_sorted_set) / len(ground_truth_set)



