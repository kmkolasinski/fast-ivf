from collections import defaultdict
from typing import Iterator, Union, Optional, Literal

import numpy as np
from numba import njit, prange
from tqdm import tqdm
from numba.typed import List as NbList

MetricType = Literal["cosine", "l2"]


class MiniBatchKMeans:
    def __init__(
        self,
        num_centroids: int,
        batch_size: int,
        history_size: int = 10,
        max_steps: int = 5000,
        tol: float = 0.0001,
        min_max_ratio_threshold: float = 0.05,
        metric: MetricType = "cosine",
    ):
        self.num_centroids = num_centroids
        self.batch_size = batch_size
        self.history_size = history_size
        self.max_steps = max_steps
        self.tol = tol
        self.min_max_ratio_threshold = min_max_ratio_threshold
        self.metric = metric

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

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

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

        centroids = np.array(centroids)
        centroids_counts = np.ones(self.num_centroids, dtype=np.float32)
        return centroids, centroids_counts

    def to_average_centroids(
        self, centroids_sums: np.ndarray, centroids_counts: np.ndarray
    ) -> np.ndarray:
        if self.metric == "cosine":
            centroids = centroids_sums / centroids_counts[..., None]
            centroids = normalize(centroids)
            return np.transpose(centroids)
        elif self.metric == "l2":
            centroids = centroids_sums / centroids_counts[..., None]
            return np.transpose(centroids)
        else:
            raise NotImplementedError(f"Unknown metric {self.metric}")

    def get_centroids_assignments(self) -> dict[int, list[int]]:
        centroid_assignments = defaultdict(list)
        for i, c in enumerate(self._labels):
            centroid_assignments[int(c)].append(i)
        return centroid_assignments

    def add(self, vectors: np.ndarray):
        assignments = []
        batches = iterate_vectors(vectors, self.batch_size)
        num_batches = vectors.shape[0] // self.batch_size
        for batch in tqdm(batches, "Assigning", total=num_batches):
            assignments.append(self.predict(batch))
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
            indices = centroids_labels[i]
            if len(indices) == 0:
                centroids_indices.append(np.zeros((0,), dtype=np.int64))
            else:
                centroids_indices.append(np.array(centroids_labels[i]))
        self._centroids_indices = centroids_indices

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        return compute_assignments(self._centroids, vectors, self.metric)[1]

    def predict_scores(self, vectors: np.ndarray) -> np.ndarray:
        return compute_distances(self._centroids, vectors, self.metric)

    def train(self, data: Union[np.ndarray, Iterator[np.ndarray]]):
        if isinstance(data, np.ndarray):
            data = to_train_generator(data, self.batch_size)

        self._history = []
        centroids_sums, centroids_counts = self.init_centroids(data)
        self._centroids = self.to_average_centroids(centroids_sums, centroids_counts)

        centroids_assignments = []
        batches_history = []

        hist_ema = 0.95
        history = [1.0]

        for _ in (pbar := tqdm(range(self.max_steps))):
            batch = next(data).copy()

            assignments = self.predict(batch)

            update_assignments_add(centroids_sums, centroids_counts, batch, assignments)
            centroids_assignments.append(assignments)
            batches_history.append(batch)

            if len(centroids_assignments) > self.history_size:
                old_assignments = centroids_assignments.pop(0)
                old_batch = batches_history.pop(0)

                new_assignments = self.predict(old_batch)

                ratio = (old_assignments != new_assignments).mean(-1).max()
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

            self._centroids = self.to_average_centroids(
                centroids_sums, centroids_counts
            )

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


class SubvectorsMiniBatchKMeans:
    def __init__(
        self,
        num_centroids: int,
        num_subvectors: int,
        batch_size: int,
        history_size: int = 10,
        max_steps: int = 1000,
        tol: float = 1e-5,
        min_max_ratio_threshold: float = 0.05,
        metric: MetricType = "cosine",
    ):
        self.num_centroids = num_centroids
        self.num_subvectors = num_subvectors
        self.batch_size = batch_size
        self.history_size = history_size
        self.max_steps = max_steps
        self.tol = tol
        self.min_max_ratio_threshold = min_max_ratio_threshold
        self.metric = metric

        self._centroids: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._history = []

    @property
    def num_vectors(self) -> int:
        return self._vectors.shape[0]

    def init_centroids(self, data: Iterator[np.ndarray]):
        centroids = []
        while len(centroids) < self.num_centroids:
            batch = next(data).copy()
            for vector in batch:
                if len(centroids) < self.num_centroids:
                    centroids.append(np.split(vector, self.num_subvectors))
                else:
                    break

        centroids = np.array(centroids)
        centroids = np.transpose(centroids, (1, 0, 2))
        centroids_counts = np.ones(
            (self.num_subvectors, self.num_centroids), dtype=np.float32
        )
        return centroids, centroids_counts

    def to_average_centroids(
        self, centroids_sums: np.ndarray, centroids_counts: np.ndarray
    ) -> np.ndarray:
        if self.metric == "cosine":
            centroids = centroids_sums / (centroids_counts[..., None] + 1e-5)
            centroids = normalize(centroids)
            centroids = np.transpose(centroids, (0, 2, 1))
            return centroids
        elif self.metric == "l2":
            centroids = centroids_sums / (centroids_counts[..., None] + 1e-5)
            centroids = np.transpose(centroids, (0, 2, 1))
            return centroids
        else:
            raise NotImplementedError(f"Unknown metric {self.metric}")

    def add(self, vectors: np.ndarray):
        assignments = []
        batches = iterate_vectors(vectors, self.batch_size)
        num_batches = vectors.shape[0] // self.batch_size
        for batch in tqdm(batches, "Assigning", total=num_batches):
            assignments.append(self.predict_labels(batch))
        assignments = np.concatenate(assignments, axis=1)
        assignments = np.transpose(assignments)

        if self._labels is None:
            self._labels = assignments
            self._vectors = vectors
        else:
            self._labels = np.concatenate([self._labels, assignments], axis=1)
            self._vectors = np.concatenate([self._vectors, vectors])

    def predict_all(self, vectors: np.ndarray):
        batch = np.reshape(vectors, (vectors.shape[0], self.num_subvectors, -1))
        batch = np.transpose(batch, (1, 0, 2))
        return batch, batch_compute_assignments(self._centroids, batch, self.metric)

    def predict_labels(self, vectors: np.ndarray) -> np.ndarray:
        _, labels = self.predict_all(vectors)[1]
        return labels

    def predict_scores(self, vectors: np.ndarray) -> np.ndarray:
        scores, _ = self.predict_all(vectors)[1]
        return scores

    def train(self, data: Union[np.ndarray, Iterator[np.ndarray]]):
        if isinstance(data, np.ndarray):
            data = to_train_generator(data, self.batch_size)

        self._history = []
        centroids_sums, centroids_counts = self.init_centroids(data)
        self._centroids = self.to_average_centroids(centroids_sums, centroids_counts)

        centroids_assignments = []
        batches_history = []

        hist_ema = 0.95
        history = [1.0]

        for _ in (pbar := tqdm(range(self.max_steps))):
            batch = next(data).copy()

            batch_split, (_, assignments) = self.predict_all(batch)

            batch_update_assignments_add(
                centroids_sums, centroids_counts, batch_split, assignments
            )
            centroids_assignments.append(assignments)
            batches_history.append(batch)

            if len(centroids_assignments) > self.history_size:
                old_assignments = centroids_assignments.pop(0)
                old_batch = batches_history.pop(0)

                old_batch_split, (_, new_assignments) = self.predict_all(old_batch)

                ratio = (old_assignments != new_assignments).mean()
                ratio = ratio * (1 - hist_ema) + history[-1] * hist_ema

                pbar.set_description(f"Convergence delta = {ratio:.5f}")
                history.append(ratio)
                if ratio < self.tol:
                    break

                batch_update_assignments_sub(
                    centroids_sums, centroids_counts, old_batch_split, old_assignments
                )
                batch_update_assignments_add(
                    centroids_sums, centroids_counts, old_batch_split, new_assignments
                )
            self.equalize_centroids(centroids_sums, centroids_counts)

            self._centroids = self.to_average_centroids(
                centroids_sums, centroids_counts
            )

        self._history = history
        return history

    def equalize_centroids(self, batch_centroids_sums, batch_centroids_counts):
        for centroids_sums, centroids_counts in zip(
            batch_centroids_sums, batch_centroids_counts
        ):
            max_count = np.max(centroids_counts)
            min_count = np.min(centroids_counts)
            min_max_ratio = min_count / max_count
            if min_max_ratio > self.min_max_ratio_threshold:
                continue

            max_index = np.argmax(centroids_counts)
            min_index = np.argmin(centroids_counts)
            tmp_max_sum = centroids_sums[max_index].copy() / 2
            tmp_max_count = centroids_counts[max_index].copy() / 2
            centroids_sums[max_index] /= 2
            centroids_counts[max_index] /= 2
            centroids_sums[min_index] += tmp_max_sum
            centroids_counts[min_index] += tmp_max_count


def compute_distances(centroids: np.ndarray, batch: np.ndarray, metric: MetricType):
    if metric == "cosine":
        return batch @ centroids
    elif metric == "l2":
        return l2_distance_matrix(centroids, batch)
    raise NotImplementedError(f"Unknown metric {metric}")


def compute_assignments(centroids: np.ndarray, batch: np.ndarray, metric: MetricType):
    values = compute_distances(centroids, batch, metric)
    if metric == "cosine":
        return values, np.argmax(values, axis=-1)
    elif metric == "l2":
        return values, np.argmin(values, axis=-1)
    raise NotImplementedError(f"Unknown metric {metric}")


def batch_compute_distances(
    centroids: np.ndarray, batch: np.ndarray, metric: MetricType
):
    if metric == "cosine":
        return batch @ centroids
    elif metric == "l2":
        return batch_l2_distance_matrix(centroids, batch)
    raise NotImplementedError(f"Unknown metric {metric}")


def batch_compute_assignments(
    centroids: np.ndarray, batch: np.ndarray, metric: MetricType
):
    values = batch_compute_distances(centroids, batch, metric)
    if metric == "cosine":
        return values, np.argmax(values, axis=-1)
    elif metric == "l2":
        return values, np.argmin(values, axis=-1)
    raise NotImplementedError(f"Unknown metric {metric}")


@njit(fastmath=True)
def l2_distance_matrix(centroids: np.ndarray, batch: np.ndarray) -> np.ndarray:
    sqnorm1 = np.sum(np.square(batch), 1)[:, None]
    sqnorm2 = np.sum(np.square(centroids), 0)[None, :]
    innerprod = batch @ centroids
    return sqnorm1 + sqnorm2 - 2.0 * innerprod


def batch_l2_distance_matrix(centroids: np.ndarray, batch: np.ndarray) -> np.ndarray:
    sqnorm1 = np.sum(np.square(batch), 2)[..., None]
    sqnorm2 = np.sum(np.square(centroids), 1)[:, None, :]
    innerprod = batch @ centroids
    return sqnorm1 + sqnorm2 - 2.0 * innerprod


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-5)


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


@njit(parallel=True)
def batch_update_assignments_add(
    centroids_sums: np.ndarray,
    centroids_counts: np.ndarray,
    vectors: np.ndarray,
    assignments: np.ndarray,
):
    for p in prange(assignments.shape[0]):
        update_assignments_add(
            centroids_sums[p], centroids_counts[p], vectors[p], assignments[p]
        )


@njit(parallel=True)
def batch_update_assignments_sub(
    centroids_sums: np.ndarray,
    centroids_counts: np.ndarray,
    vectors: np.ndarray,
    assignments: np.ndarray,
):
    for p in prange(assignments.shape[0]):
        update_assignments_sub(
            centroids_sums[p], centroids_counts[p], vectors[p], assignments[p]
        )


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
