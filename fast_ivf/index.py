from fast_ivf.core import approximated_search, rescore_queries, approximated_search_pq
from fast_ivf.kmeans import MiniBatchKMeans, SubvectorsMiniBatchKMeans

from numba.typed import List as NbList
import numpy as np
from typing import Iterator, Union


class FastIVF:
    def __init__(
        self,
        ndim: int,
        nlist: int,
        nprobe: int = 10,
        ratio_threshold: float = 0.0001,
        kmeans_batch_size: int = 512,
        kmeans_tol: float = 0.00005,
        kmeans_max_steps: int = 1000,
    ):
        self.ndim = ndim
        self.nlist = nlist
        self.nprobe = nprobe
        self.ratio_threshold = ratio_threshold

        self.kmeans = MiniBatchKMeans(
            nlist,
            batch_size=kmeans_batch_size,
            tol=kmeans_tol,
            max_steps=kmeans_max_steps,
        )

    def train(self, data: Union[np.ndarray, Iterator[np.ndarray]]):
        self.kmeans.train(data)
        self.kmeans.add(data)

    def search(
        self,
        query: np.ndarray,
        k: int = 100,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        centroids_scores = self.kmeans.predict_scores(query)

        top_scores, top_indices = approximated_search(
            query,
            centroids_scores,
            self.kmeans.centroids_indices,
            self.kmeans.vectors,
            nprobe=self.nprobe,
            topk=k,
            ratio_threshold=self.ratio_threshold,
        )

        return top_scores, top_indices


class CompressedFastIVF:
    def __init__(
        self,
        ndim: int,
        nlist: int,
        compression_ndim: int,
        nprobe: int = 10,
        ratio_threshold: float = 0.0001,
        kmeans_batch_size: int = 512,
        kmeans_tol: float = 0.00005,
        kmeans_max_steps: int = 1000,
        compressor_batch_size: int = 128,
        compressor_steps_per_epoch: int = 5000,
        compressor_epochs: int = 10,
        rescore_num_samples: int = -1,
    ):
        self.ndim = ndim
        self.nlist = nlist
        self.compression_ndim = compression_ndim
        self.nprobe = nprobe
        self.ratio_threshold = ratio_threshold
        self.rescore_num_samples = rescore_num_samples

        self.kmeans = MiniBatchKMeans(
            nlist,
            batch_size=kmeans_batch_size,
            tol=kmeans_tol,
            max_steps=kmeans_max_steps,
        )
        from fast_ivf.tf_compressor import TFCentroidCompressor

        self.compressor = TFCentroidCompressor(
            ndim=ndim,
            num_centroids=nlist,
            bottleneck=compression_ndim,
            steps_per_epoch=compressor_steps_per_epoch,
            batch_size=compressor_batch_size,
            epochs=compressor_epochs,
        )

    def train(self, data: Union[np.ndarray, Iterator[np.ndarray]]):
        self.kmeans.train(data)
        self.kmeans.add(data)

        if self.compressor.steps_per_epoch == -1:
            steps_per_epoch = self.kmeans.num_vectors // self.compressor.batch_size
            self.compressor.steps_per_epoch = steps_per_epoch

        dataset_iterator_fn = self.kmeans.get_dataset_iterator
        self.compressor.train(dataset_iterator_fn)
        self.compressor.add(dataset_iterator_fn)

    def search(
        self,
        query: np.ndarray,
        k: int = 100,
        rescore: bool = False,
        sort: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        query_projected, _ = self.compressor.predict(query)
        centroids_scores = self.kmeans.predict_scores(query)

        top_scores, top_indices = approximated_search(
            query_projected,
            centroids_scores,
            self.kmeans.centroids_indices,
            self.compressor.vectors,
            nprobe=self.nprobe,
            topk=k,
            ratio_threshold=self.ratio_threshold,
        )

        if rescore:
            top_scores_nb = NbList(top_scores)
            top_indices_nb = NbList(top_indices)
            vectors = self.kmeans.vectors

            top_scores, top_indices = rescore_queries(
                query,
                top_indices_nb,
                top_scores_nb,
                vectors,
                sort=sort,
                rescore_num_samples=self.rescore_num_samples,
            )
            return list(top_scores), list(top_indices)

        return top_scores, top_indices


class FastIVFPQ:
    def __init__(
        self,
        ndim: int,
        nlist: int,
        pq_num_centroids: int = 32,
        pq_num_subvectors: int = 16,
        nprobe: int = 10,
        ratio_threshold: float = 0.0001,
        kmeans_batch_size: int = 512,
        kmeans_pq_batch_size: int = 64,
        kmeans_tol: float = 0.0001,
        kmeans_max_steps: int = 1000,
        kmeans_pq_max_steps: int = 10000,
        rescore_num_samples: int = -1,
    ):
        self.ndim = ndim
        self.nlist = nlist
        self.pq_num_centroids = pq_num_centroids
        self.pq_num_subvectors = pq_num_subvectors
        self.nprobe = nprobe
        self.ratio_threshold = ratio_threshold
        self.rescore_num_samples = rescore_num_samples

        self.kmeans = MiniBatchKMeans(
            num_centroids=nlist,
            batch_size=kmeans_batch_size,
            tol=kmeans_tol,
            max_steps=kmeans_max_steps,
        )

        self.pq_kmeans = SubvectorsMiniBatchKMeans(
            num_centroids=pq_num_centroids,
            num_subvectors=pq_num_subvectors,
            batch_size=kmeans_pq_batch_size,
            tol=kmeans_tol,
            max_steps=kmeans_pq_max_steps,
            metric="l2",
        )

    def train(self, data: Union[np.ndarray, Iterator[np.ndarray]]):
        self.kmeans.train(data)
        self.kmeans.add(data)

        self.pq_kmeans.train(data)
        self.pq_kmeans.add(data)

    def search(
        self,
        query: np.ndarray,
        k: int = 100,
        rescore: bool = False,
        sort: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        centroids_scores = self.kmeans.predict_scores(query)
        pq_centroid_scores = self.pq_kmeans.predict_scores(query)
        pq_centroid_scores = np.transpose(pq_centroid_scores, (1, 0, 2))
        pq_labels_to_centroids = self.pq_kmeans._labels
        top_scores, top_indices = approximated_search_pq(
            centroids_scores,
            self.kmeans.centroids_indices,
            pq_centroid_scores,
            pq_labels_to_centroids,
            nprobe=self.nprobe,
            topk=k,
            ratio_threshold=self.ratio_threshold,
        )
        if rescore:
            top_scores_nb = NbList(top_scores)
            top_indices_nb = NbList(top_indices)
            vectors = self.kmeans.vectors

            top_scores, top_indices = rescore_queries(
                query,
                top_indices_nb,
                top_scores_nb,
                vectors,
                sort=sort,
                rescore_num_samples=self.rescore_num_samples,
            )
            return list(top_scores), list(top_indices)

        return top_scores, top_indices
