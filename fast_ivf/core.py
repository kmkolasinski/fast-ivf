from typing import Union

import numba as nb
from numba import prange
from numba.typed import List
import numpy as np


@nb.njit(fastmath=True, parallel=True)
def get_rows_to_centroids(indices: nb.int32[:, :], nlist: int):
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
    query_points_scores,
    query_points_indices,
    sort: bool,
    topk: int,
    ratio_threshold: float = 0.0,
):
    num_queries = len(query_points_scores)
    for i in prange(num_queries):
        scores = query_points_scores[i]
        sort_indices, new_scores = top_k(
            scores, k=topk, sort=sort, ratio_threshold=ratio_threshold
        )
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


@nb.njit
def approximated_search(
    queries_projected: nb.float32[:, :],
    queries_centroids_predictions: nb.float32[:, :],
    faiss_clusters: list[nb.int32[:]],
    train_embeddings_projected: nb.float32[:, :],
    nprobe: int = 5,
    topk: int = 100,
    ratio_threshold: float = 0.0,
    sort: bool = True,
):
    nlist = queries_centroids_predictions.shape[1]
    indices, _ = batched_top_k(
        queries_centroids_predictions,
        nprobe,
        sort=False,
        ratio_threshold=ratio_threshold,
    )
    rows_per_centroid = get_rows_to_centroids(indices, nlist)

    num_queries = queries_projected.shape[0]
    query_points_count = np.zeros((num_queries,), dtype=np.int32)

    for centroid_id, query_indices in enumerate(rows_per_centroid):
        index_indices = faiss_clusters[centroid_id]
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

    for centroid_id, query_indices in enumerate(rows_per_centroid):
        index_indices = faiss_clusters[centroid_id]
        query_proj = queries_projected[query_indices]
        train_proj = train_embeddings_projected[index_indices]

        scores = query_proj @ train_proj.T
        append_scores_and_indices(
            query_indices,
            index_indices,
            scores,
            query_points_offset,
            query_points_scores,
            query_points_indices,
        )

    select_topk(query_points_scores, query_points_indices, topk=topk, sort=sort)

    return query_points_scores, query_points_indices
