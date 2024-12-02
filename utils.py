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
