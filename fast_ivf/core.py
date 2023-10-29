from typing import Union

import numba as nb
from numba import prange
from numba.typed import List
import numpy as np


@nb.njit(fastmath=True, parallel=True)
def get_centroids_to_rows(indices: nb.int32[:, :], nlist: int):
    per_row = List.empty_list(nb.int32[:])
    for i in range(nlist):
        per_row.append(indices[0, :1])

    for i in prange(nlist):
        per_row[i] = np.nonzero(indices == i)[0].astype(np.int32)
    return per_row


@nb.njit(fastmath=True, parallel=True)
def rescore_queries(
    query: nb.float32[:, :],
    query_indices: list[nb.int32[:]],
    query_scores: list[nb.int32[:]],
    vectors: nb.float32[:, :],
    sort: bool = False,
    rescore_num_samples: int = 20,
):
    exact_top_indices = List.empty_list(nb.int32[:])
    exact_top_scores = List.empty_list(nb.float32[:])

    num_queries = query.shape[0]
    rescore_ratio = 1 / max(rescore_num_samples, 1)

    for i in range(num_queries):
        indices = query_indices[i]
        exact_top_indices.append(indices[:1])
        exact_top_scores.append(np.array([0.0], dtype=np.float32))

    for i in prange(num_queries):
        indices = query_indices[i]
        approx_scores = query_scores[i]

        if 0 < rescore_num_samples < len(indices):
            stride = max(int(len(indices) * rescore_ratio), 1)
            exact_scores = vectors[indices[::stride]] @ query[i]
            m, c = fit_linear(exact_scores, approx_scores[::stride])
            exact_scores = np.clip(m * approx_scores + c, 0, 1)
        else:
            vectors_slice = vectors[indices]
            exact_scores = vectors_slice @ query[i]

        if sort:
            sort_indices = np.argsort(exact_scores)[::-1]
            exact_top_indices[i] = indices[sort_indices]
            exact_top_scores[i] = exact_scores[sort_indices]
        else:
            exact_top_indices[i] = indices
            exact_top_scores[i] = exact_scores

    return exact_top_scores, exact_top_indices


@nb.njit(fastmath=True)
def fit_linear(y, x):
    A = np.stack((x, np.ones_like(x)), axis=1)
    m, c = np.linalg.lstsq(A, y)[0]
    return m, c


@nb.njit(fastmath=True)
def top_k(scores: nb.float32[:], k: int, sort: bool, ratio_threshold: float = 0.0):
    MIN_VALUE = -1e6
    sorted_indices = (MIN_VALUE * np.ones((k,))).astype(np.float32)
    minimum_index = 0
    minimum_index_value = nb.float32(MIN_VALUE)

    for value in scores:
        if value > minimum_index_value:
            sorted_indices[minimum_index] = value
            minimum_index = sorted_indices.argmin()
            minimum_index_value = sorted_indices[minimum_index]

    top_indices = (scores >= minimum_index_value).nonzero()[0][-k:]
    top_scores = scores[top_indices]

    if ratio_threshold > 0.0:
        max_score = np.abs(np.max(top_scores))
        top_indices = np.where(
            top_scores / max_score < ratio_threshold, -1, top_indices
        )

    if sort:
        si = np.argsort(top_scores)[::-1]
        return top_indices[si], top_scores[si]
    else:
        return top_indices, top_scores


@nb.njit(fastmath=True)
def top_k_numpy(scores: nb.float32[:], k: int, is_score: bool = True):
    if is_score:
        top_indices = np.argsort(scores)[::-1][:k]
    else:
        top_indices = np.argsort(scores)[:k]
    return top_indices, scores[top_indices]


@nb.njit(parallel=True)
def append_scores_and_indices(
    query_indices,
    index_indices,
    scores,
    query_points_offset,
    query_points_scores,
    query_points_indices,
):
    num_indices = query_indices.shape[0]
    for i in prange(num_indices):
        qi = query_indices[i]
        count = scores[i].shape[0]
        offset = query_points_offset[qi]
        query_points_scores[qi][offset : offset + count] = scores[i]
        query_points_indices[qi][offset : offset + count] = index_indices
        query_points_offset[qi] += count


@nb.njit
def pad_to(array: np.ndarray, k: int, value: Union[float, int] = -1):
    if len(array) < k:
        dtype = array.dtype
        to_pad = k - len(array)
        values_to_pad = np.full(to_pad, value, dtype=dtype)
        return np.concatenate((array, values_to_pad))
    return array


@nb.njit(parallel=True)
def select_topk(
    query_points_scores: nb.float32[:],
    query_points_indices: nb.int32[:],
    topk: int,
    is_score: bool = True,
):
    num_queries = len(query_points_scores)
    for i in prange(num_queries):
        scores = query_points_scores[i]

        if topk / len(scores) < 0.3:
            if is_score:
                sort_indices, new_scores = top_k(scores, k=topk, sort=True)
            else:
                sort_indices, new_scores = top_k(-scores, k=topk, sort=True)
                new_scores = -new_scores
        else:
            sort_indices, new_scores = top_k_numpy(scores, k=topk, is_score=is_score)

        new_indices = query_points_indices[i][sort_indices]
        if len(new_indices) < topk:
            new_indices = pad_to(new_indices, topk)
            new_scores = pad_to(new_scores, topk)
        query_points_indices[i] = new_indices
        query_points_scores[i] = new_scores


@nb.njit(fastmath=True, parallel=True)
def batched_top_k(
    scores: nb.float32[:, :], k: int, sort: bool = True, ratio_threshold: float = 0.0
):
    top_indices = np.zeros((scores.shape[0], k), dtype=np.int32)
    top_scores = np.zeros((scores.shape[0], k), dtype=np.float32)
    for i in prange(scores.shape[0]):
        top_indices[i], top_scores[i] = top_k(
            scores[i], k, sort=sort, ratio_threshold=ratio_threshold
        )
    return top_indices, top_scores


@nb.njit(fastmath=True, parallel=True)
def batched_top_k_numpy(scores: nb.float32[:, :], k: int, is_score: bool = True):
    top_indices = np.zeros((scores.shape[0], k), dtype=np.int32)
    top_scores = np.zeros((scores.shape[0], k), dtype=np.float32)
    for i in prange(scores.shape[0]):
        top_indices[i], top_scores[i] = top_k_numpy(scores[i], k, is_score)
    return top_indices, top_scores


@nb.njit(fastmath=True, parallel=True)
def compute_partial_scores(
    rows_per_centroid: list[nb.int32[:]],
    centroids_to_indices: list[nb.int32[:]],
    queries_projected: nb.float32[:, :],
    train_embeddings_projected: nb.int32[:, :],
):
    num_centroids = len(rows_per_centroid)
    partial_scores = []

    for centroid_id in range(num_centroids):
        partial_scores.append(np.zeros((1, 1), dtype=np.float32))

    for centroid_id in prange(num_centroids):
        query_indices = rows_per_centroid[centroid_id]
        train_indices = centroids_to_indices[centroid_id]

        query_proj = queries_projected[query_indices]
        train_proj = train_embeddings_projected[train_indices]

        scores = query_proj @ train_proj.T
        partial_scores[centroid_id] = scores
    return partial_scores


@nb.njit
def approximated_search(
    query_embeddings_projected: nb.float32[:, :],
    query_centroids_predictions: nb.float32[:, :],
    centroids_to_indices: list[nb.int32[:]],
    train_embeddings_projected: nb.float32[:, :],
    nprobe: int = 5,
    topk: int = 100,
    ratio_threshold: float = 0.0,
):
    nlist = query_centroids_predictions.shape[1]
    indices, _ = batched_top_k(
        query_centroids_predictions,
        nprobe,
        sort=False,
        ratio_threshold=ratio_threshold,
    )
    rows_per_centroid = get_centroids_to_rows(indices, nlist)

    num_queries = query_embeddings_projected.shape[0]
    query_points_count = np.zeros((num_queries,), dtype=np.int32)

    for centroid_id, query_indices in enumerate(rows_per_centroid):
        index_indices = centroids_to_indices[centroid_id]
        for i, qi in enumerate(query_indices):
            query_points_count[qi] += len(index_indices)

    query_points_scores = []
    query_points_indices = []
    query_points_offset = np.zeros((num_queries,), dtype=np.int32)

    for centroid_id in range(num_queries):
        count = query_points_count[centroid_id]
        query_points_scores.append(np.zeros((count,), dtype=np.float32))
        query_points_indices.append(np.zeros((count,), dtype=np.int32))
        query_points_offset[centroid_id] = 0

    partial_scores = compute_partial_scores(
        rows_per_centroid,
        centroids_to_indices,
        query_embeddings_projected,
        train_embeddings_projected,
    )

    for centroid_id, query_indices in enumerate(rows_per_centroid):
        train_indices = centroids_to_indices[centroid_id]
        scores = partial_scores[centroid_id]

        append_scores_and_indices(
            query_indices,
            train_indices,
            scores,
            query_points_offset,
            query_points_scores,
            query_points_indices,
        )

    select_topk(query_points_scores, query_points_indices, topk=topk)

    return query_points_scores, query_points_indices


@nb.njit(fastmath=True, parallel=True)
def compute_partial_scores_pq(
    rows_per_centroid: list[nb.int32[:]],
    centroids_to_indices: list[nb.int32[:]],
    pq_centroid_scores: nb.float32[:, :, :],
    pq_labels_to_centroids: nb.int32[:, :],
):
    num_centroids = len(rows_per_centroid)
    partial_scores = []

    for centroid_id in range(num_centroids):
        partial_scores.append(np.zeros((1, 1), dtype=np.float32))

    for centroid_id in prange(num_centroids):
        query_indices = rows_per_centroid[centroid_id]
        train_indices = centroids_to_indices[centroid_id]

        query_scores = pq_centroid_scores[query_indices]
        train_labels = pq_labels_to_centroids[train_indices]

        scores = compute_approx_scores_pq(query_scores, train_labels)

        partial_scores[centroid_id] = scores
    return partial_scores


@nb.njit
def approximated_search_pq(
    centroids_scores: nb.float32[:, :],
    centroids_to_indices: list[nb.int32[:]],
    pq_centroid_scores: nb.float32[:, :, :],
    pq_labels_to_centroids: nb.int32[:, :],
    nprobe: int = 5,
    topk: int = 100,
    ratio_threshold: float = 0.0,
):
    nlist = centroids_scores.shape[1]
    indices, _ = batched_top_k(
        centroids_scores,
        nprobe,
        sort=False,
        ratio_threshold=ratio_threshold,
    )
    rows_per_centroid = get_centroids_to_rows(indices, nlist)

    num_queries = centroids_scores.shape[0]
    query_points_count = np.zeros((num_queries,), dtype=np.int32)

    for centroid_id, query_indices in enumerate(rows_per_centroid):
        index_indices = centroids_to_indices[centroid_id]
        for i, qi in enumerate(query_indices):
            query_points_count[qi] += len(index_indices)

    query_points_scores = []
    query_points_indices = []
    query_points_offset = np.zeros((num_queries,), dtype=np.int32)

    for centroid_id in range(num_queries):
        count = query_points_count[centroid_id]
        query_points_scores.append(np.zeros((count,), dtype=np.float32))
        query_points_indices.append(np.zeros((count,), dtype=np.int32))
        query_points_offset[centroid_id] = 0

    partial_scores = compute_partial_scores_pq(
        rows_per_centroid,
        centroids_to_indices,
        pq_centroid_scores,
        pq_labels_to_centroids,
    )

    for centroid_id, query_indices in enumerate(rows_per_centroid):
        train_indices = centroids_to_indices[centroid_id]
        scores = partial_scores[centroid_id]

        append_scores_and_indices(
            query_indices,
            train_indices,
            scores,
            query_points_offset,
            query_points_scores,
            query_points_indices,
        )

    select_topk(query_points_scores, query_points_indices, topk=topk, is_score=False)

    return query_points_scores, query_points_indices


@nb.njit
def compute_approx_scores_pq(pred_distances: np.ndarray, train_labels: np.ndarray):
    num_predictions = pred_distances.shape[0]
    num_targets = train_labels.shape[0]
    num_subspaces = train_labels.shape[1]

    scores = np.zeros((num_predictions, num_targets), dtype=np.float32)

    for p in range(num_predictions):
        for t in range(num_targets):
            for s in range(num_subspaces):
                scores[p, t] += pred_distances[p, s, train_labels[t, s]]
    return scores


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-5)
