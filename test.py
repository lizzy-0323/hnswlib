import unittest
import numpy as np
from main import merge_top_k


class TestAggregation(unittest.TestCase):
    def test_merge_top_k(self):
        result_lst = [
            (np.array([[1, 2], [3, 4]]), np.array([[0.1, 0.4], [0.7, 0.8]])),
            (np.array([[5, 6], [7, 8]]), np.array([[0.5, 0.6], [0.2, 0.3]])),
        ]
        top_k = 2

        top_k_labels, top_k_distances = merge_top_k(result_lst, top_k)

        expected_labels = np.array([[1, 2], [7, 8]])
        expected_distances = np.array([[0.1, 0.4], [0.2, 0.3]])

        np.testing.assert_array_equal(top_k_labels, expected_labels)
        np.testing.assert_array_equal(top_k_distances, expected_distances)


if __name__ == "__main__":
    unittest.main()
