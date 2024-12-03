from typing import Literal
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian


def partition_data(data: np.ndarray, num_partitions: int) -> list:
    print("Parition number: ", num_partitions)
    num, dim = data.shape
    partition_size = num // num_partitions
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        partitions.append(data[start:end])
    return partitions


def sample_data(data: np.ndarray, num_samples: int, method="random") -> np.ndarray:
    num, dim = data.shape
    indices = np.array([])
    if method == "random":
        indices = np.random.choice(num, num_samples, replace=False)
    elif method == "uniform":
        indices = np.linspace(0, num - 1, num_samples, dtype=int)
    else:
        raise ValueError("Invalid sampling method")
    return data[indices]


def kmeans_partition(
    data: np.ndarray, num_clusters: int, space: Literal["l2", "cosine", "ip"] = "l2"
):
    """
    Partition the data using kmeans clustering
    :return: buckets and corresponding centroid: [(bucket1, centroid1), (bucket2, centroid2), ...]
    """
    # TODO: Sklearn only supports l2 distance, need to implement other distance metrics
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
    kmeans.fit(data)
    labels = kmeans.labels_  # num_samples * 1, min: 0, max: num_clusters - 1
    centroids = kmeans.cluster_centers_
    buckets = []
    for i in range(num_clusters):
        bucket = data[labels == i]
        buckets.append((bucket, centroids[i]))
    return buckets


def partition_buckets_spectral(
    buckets: list,
    n_partitions: int,
    space: Literal["l2", "cosine", "ip"] = "l2",
):
    """
    Partition buckets
    For now only spectral partitioning is implemented, not graph partitioning
    :param buckets: [(bucket1, centroid1), (bucket2, centroid2), ...]
    :return: partitions: [[(bucket1, centroid1), (bucket2, centroid2), ...], ...]
    """
    centroid_array = np.array([bucket[1] for bucket in buckets])
    dim = centroid_array.shape[1]
    if space == "l2":
        distance_matrix = np.linalg.norm(
            centroid_array[:, np.newaxis, :] - centroid_array[np.newaxis, :, :], axis=-1
        )
        print("Max distance: ", np.max(distance_matrix))
        similarity_matrix = 1 / (1 + distance_matrix)
    elif space == "cosine":
        norms = np.linalg.norm(centroid_array, axis=1, keepdims=True)
        normalized_vectors = centroid_array / norms
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    else:
        raise ValueError("Error: Not implemented yet")
    L = laplacian(similarity_matrix, normed=True)
    eigvals, eigvecs = np.linalg.eig(L)
    eigvecs = eigvecs[:, :n_partitions]
    kmeans = KMeans(n_clusters=n_partitions, n_init="auto")
    labels = kmeans.fit_predict(eigvecs)
    print(labels)
    partitions = []
    for i in range(n_partitions):
        partition = []
        for j in range(len(labels)):
            if labels[j] == i:
                partition.append(buckets[j])
        partitions.append(partition)
    return partitions

def partition_buckets_kmeans( buckets: list, n_partitions: int):
    """
    Partition buckets using kmeans clustering
    """
    centroid_array = np.array([bucket[1] for bucket in buckets])
    kmeans = KMeans(n_clusters=n_partitions, n_init="auto")
    labels = kmeans.fit_predict(centroid_array)
    partitions = []
    for i in range(n_partitions):
        partition = []
        for j in range(len(labels)):
            if labels[j] == i:
                partition.append(buckets[j])
        partitions.append(partition)
    return partitions


def validate_partitioning(partitions: list):
    """
    Validate the partitioning, by checking the average cluster centroid distance and the average cluster data distance
    """
    distances = []
    max_distances = []
    for partition in partitions:
        centroid_list = np.array([bucket[1] for bucket in partition])
        distance_matrix = np.linalg.norm(
            centroid_list[:, np.newaxis, :] - centroid_list[np.newaxis, :, :], axis=-1
        )
        distances.append(np.mean(distance_matrix))
        max_distances.append(np.max(distance_matrix))
    average_centroid_distance = np.mean(distances)
    inner_partition_max_distance = np.max(max_distances)

    # randomly sample one bucket's centroid from each partition, aclculate the average distance between them
    random_centroids = []
    for partition in partitions:
        partition_len = len(partition)
        choice = np.random.choice(partition_len)
        random_centroids.append(partition[choice][1])
    random_centroids = np.array(random_centroids)
    distance_matrix = np.linalg.norm(
        random_centroids[:, np.newaxis, :] - random_centroids[np.newaxis, :, :], axis=-1
    )
    random_centroid_avg_distance = np.mean(distance_matrix)
    random_centroid_min_distance = np.min(distance_matrix[distance_matrix > 0])
    random_centroid_max_distance = np.max(distance_matrix[distance_matrix > 0])
    print("Average centroid distance: ", average_centroid_distance)
    print("Random centroid average distance: ", random_centroid_avg_distance)
    print("Max centroid distance: ", inner_partition_max_distance)
    print("Random centroid min distance", random_centroid_min_distance)
    print("Random centroid max distance: ", random_centroid_max_distance)


if __name__ == "__main__":
    data = np.random.rand(10000, 128)
    kmeans_result = kmeans_partition(data, 1000)
    # for bucket, centroid in kmeans_result:
    #     print(bucket.shape, centroid.shape)
    print(len(kmeans_result))
    partitions = partition_buckets_kmeans(kmeans_result, 100)
    for part in partitions:
        print(len(part), end=" ")
    validate_partitioning(partitions)
