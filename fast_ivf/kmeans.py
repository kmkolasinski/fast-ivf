from collections import defaultdict
from typing import Iterator, Union, Optional

import numpy as np
from numba import njit
from tqdm import tqdm
from numba.typed import List as NbList


class MiniBatchKMeans:
    def __init__(
        self,
        num_centroids: int,
        batch_size: int,
        history_size: int = 10,
        max_steps: int = 1000,
        tol: float = 1e-5,
        min_max_ratio_threshold: float = 0.05,
    ):
        self.num_centroids = num_centroids
        self.batch_size = batch_size
        self.history_size = history_size
        self.max_steps = max_steps
        self.tol = tol
        self.min_max_ratio_threshold = min_max_ratio_threshold
        self._centroids: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._centroids_indices: Optional[NbList[np.ndarray]] = None
        self._history = []

    @property
    def num_vectors(self) -> int:
        return self._vectors.shape[0]

    @property
    def centroids_indices(self) -> NbList[np.ndarray]:
        return self._centroids_indices

    def get_vectors(self, indices: np.ndarray) -> np.ndarray:
        return self._vectors[indices]
    def init_centroids(self, data: Iterator[np.ndarray]):
        centroids = []
        while len(centroids) < self.num_centroids:
            batch = next(data).copy()
            for vector in batch:
                if len(centroids) < self.num_centroids:
                    centroids.append(vector)
                else:
                    break

        return np.array(centroids)

    def get_centroids_assignments(self) -> dict[int, list[int]]:
        centroid_assignments = defaultdict(list)
        for i, c in enumerate(self._labels):
            centroid_assignments[int(c)].append(i)
        return centroid_assignments

    def add(self, vectors: np.ndarray):

        assignments = []
        for batch in tqdm(iterate_vectors(vectors, self.batch_size), "Assigning"):
            assignments.append(compute_assignments(self._centroids, batch))
        assignments = np.concatenate(assignments)

        if self._labels is None:
            self._labels = assignments
            self._vectors = vectors
        else:
            self._labels = np.concatenate([self._labels, assignments])
            self._vectors = np.concatenate([self._vectors, vectors])

        # used in the search function
        centroids_indices = NbList()
        centroids_labels = self.get_centroids_assignments()
        for i in range(self.num_centroids):
            centroids_indices.append(np.array(centroids_labels[i]))
        self._centroids_indices = centroids_indices

    def train(self, data: Union[np.ndarray, Iterator[np.ndarray]]):
        if isinstance(data, np.ndarray):
            data = to_train_generator(data, self.batch_size)

        centroids = self.init_centroids(data)
        self._history = []
        centroids = np.array(centroids)
        centroids_sums = centroids.copy()
        centroids_counts = np.ones(self.num_centroids, dtype=np.float32)

        centroids_assignments = []
        batches_history = []

        hist_ema = 0.95
        history = [1.0]

        for _ in (pbar := tqdm(range(self.max_steps))):
            batch = next(data).copy()

            assignments = compute_assignments(centroids, batch)

            update_assignments_add(centroids_sums, centroids_counts, batch, assignments)
            centroids_assignments.append(assignments)
            batches_history.append(batch)

            if len(centroids_assignments) > self.history_size:
                old_assignments = centroids_assignments.pop(0)
                old_batch = batches_history.pop(0)

                new_assignments = compute_assignments(centroids, old_batch)

                ratio = (old_assignments != new_assignments).mean()
                ratio = ratio * (1 - hist_ema) + history[-1] * hist_ema

                pbar.set_description(f"Convergence delta = {ratio:.5f}")
                history.append(ratio)
                if ratio < self.tol:
                    break

                update_assignments_sub(
                    centroids_sums, centroids_counts, old_batch, old_assignments
                )
                update_assignments_add(
                    centroids_sums, centroids_counts, old_batch, new_assignments
                )
            self.equalize_centroids(_, centroids_sums, centroids_counts)
            centroids = to_normalized_centroids(centroids_sums, centroids_counts)

        self._centroids = centroids
        self._history = history
        return history

    def equalize_centroids(self, iteration: int, centroids_sums, centroids_counts):
        max_count = np.max(centroids_counts)
        min_count = np.min(centroids_counts)
        min_max_ratio = min_count / max_count
        if min_max_ratio > self.min_max_ratio_threshold:
            return

        max_index = np.argmax(centroids_counts)
        min_index = np.argmin(centroids_counts)
        # print(f"Equalizing centroids at {iteration}, ratio = {min_max_ratio:.3f} mixing {min_index} with {max_index}")
        tmp_max_sum = centroids_sums[max_index].copy() / 2
        tmp_max_count = centroids_counts[max_index].copy() / 2
        centroids_sums[max_index] /= 2
        centroids_counts[max_index] /= 2
        centroids_sums[min_index] += tmp_max_sum
        centroids_counts[min_index] += tmp_max_count

    def get_dataset_iterator(self, batch_size: int = 32) -> Iterator[np.ndarray]:
        indices = np.arange(self.num_vectors)
        batch_index = 0
        while True:
            batch_indices = indices[batch_index : batch_index + batch_size]
            batch_vectors = self._vectors[batch_indices]
            batch_labels = self._labels[batch_indices]
            batch_index += batch_size
            yield batch_vectors, batch_labels
            if batch_index >= self.num_vectors:
                break


@njit(fastmath=True)
def compute_assignments(centroids: np.ndarray, batch: np.ndarray) -> np.ndarray:
    scores = batch @ centroids.T
    assignments = np.argmax(scores, axis=-1)
    return assignments


def to_normalized_centroids(
    centroids_sums: np.ndarray, centroids_counts: np.ndarray
) -> np.ndarray:
    centroids = centroids_sums / centroids_counts[:, None]
    return centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)


@njit(fastmath=True)
def update_assignments_add(
    centroids_sums: np.ndarray,
    centroids_counts: np.ndarray,
    vectors: np.ndarray,
    assignments: np.ndarray,
):
    for p in range(assignments.shape[0]):
        i = assignments[p]
        centroids_sums[i] += vectors[p]
        centroids_counts[i] += 1


@njit(fastmath=True)
def update_assignments_sub(
    centroids_sums: np.ndarray,
    centroids_counts: np.ndarray,
    vectors: np.ndarray,
    assignments: np.ndarray,
):
    for p in range(assignments.shape[0]):
        i = assignments[p]
        centroids_sums[i] -= vectors[p]
        centroids_counts[i] -= 1


def to_train_generator(data: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    num_vectors = data.shape[0]
    while True:
        batch_indices = np.random.randint(0, num_vectors, size=batch_size)
        batch_vectors = data[batch_indices]
        yield batch_vectors


def iterate_vectors(vectors: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    num_vectors = vectors.shape[0]
    indices = np.arange(num_vectors)
    batch_index = 0
    while True:
        batch_indices = indices[batch_index : batch_index + batch_size]
        batch_vectors = vectors[batch_indices]
        batch_index += batch_size
        yield batch_vectors
        if batch_index >= num_vectors:
            break
