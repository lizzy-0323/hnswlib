import numpy as np


def calculate_recall(ground_truth: np.ndarray, labels: np.ndarray, top_k: int) -> float:
    recall = 0
    for real_candidate, candidate in zip(ground_truth, labels):
        recall += len(set(real_candidate) & set(candidate[:top_k])) / top_k
    return recall / len(ground_truth)
