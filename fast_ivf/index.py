from fast_ivf.core import approximated_search, rescore_queries
from fast_ivf.kmeans import MiniBatchKMeans
from fast_ivf.tf_compressor import TFCentroidCompressor
from numba.typed import List as NbList
import numpy as np
from typing import Iterator, Union


class FastIVF:
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
        compressor_epochs: int = 5,
        rescore_num_samples: int = 20,
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
        k: int = 10,
        rescore: bool = False,
        sort: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        query_projected, centroids_scores = self.compressor.predict(query)

        top_scores, top_indices = approximated_search(
            query_projected,
            centroids_scores,
            self.kmeans.centroids_indices,
            self.compressor.vectors,
            nprobe=self.nprobe,
            topk=k,
            ratio_threshold=self.ratio_threshold,
            sort=sort,
        )
        if rescore:

            top_scores_nb = NbList(top_scores)
            top_indices_nb = NbList(top_indices)
            vectors = self.kmeans._vectors

            return rescore_queries(
                query,
                top_indices_nb,
                top_scores_nb,
                vectors,
                sort=sort,
                rescore_num_samples=self.rescore_num_samples,
            )

        return top_scores, top_indices
