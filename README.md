# FastIVF

Efficient implementation of IVF Index with numpy and numba

## Installation

* Install numpy and numba from conda to use intel mkl libraries for linear algebra operations
* To install package run `pip install .`
* You may need to install tensorflow>=2.13, see `CompressedFastIVF` for details
* code tested with python==3.11
* see notebook [test-index](notebooks/test-index.ipynb) for Index usage examples
* see notebook [test-kmeans](notebooks/test-kmeans.ipynb) for K-means usage examples

## Features / limitations

* This is an experimental code which heavily relies on numba and numpy and may contain bugs
* IVF centroids are estimated with custom mini batch kmeans implementation
  * `MiniBatchKMeans` is used to estimate centroids of standard Inverted Index 
  * `SubspaceMiniBatchKMeans` is used to estimate centroids of Product Quantization Index
* K-means implementations support only l2 or cosine distances
* All indices currently support only cosine distance 

# Results on custom benchmark data

* Resources restricted to `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=12` which was consuming 100% in our case for fast-ivf and faiss
* Train vectors: internal ~900k vectors of dim=1024, normalized to unit length
* Test vectors: same but 40k vectors
* Hyperparams: nprobe=10, ratio_threshold=0.5, no re-scoring is used for approximated indices (for mini-batch kmeans we use repository defaults), 
for CompressionFastIVF we use compression_ndim=128 (which gives 8 times compression ratio)
* We measure recall@10, as function which checks if `exact_i is in top_indices[:10]` for each test query, then we 
average the results over all test vectors
* For faiss I used similar parameters for nlist, m, nbits etc
* Reported time is computed from average of 5 runs, divided by 40k to get the time per single query
* As we use numba internally, each Fast-Index is initialized with warmup call to compile the code
* Note: CompressedFastIVF requires to train small neural network to compress embeddings to lower dimensionality, which increases the index build time
* For both libraries each search() call was consuming all 40k vectors, to fully utilize all vectorization

| Index             | Recall@10 | Query Time (ms) | Params                                                                                   |
|-------------------|-----------|-----------------|------------------------------------------------------------------------------------------|
| FastIVF           | 0.964     | 0.100           | `nlist=1024, nprobe=10, ratio_threshold=0.5`                                             |
| Faiss IVF         | 0.968     | 1.000           | `nlist=1024, nprobe=10`                                                                  |
| FastIVFPQ         | 0.802     | 0.100           | `nlist=1024, nprobe=10, ratio_threshold=0.5, pq_num_subvectors=32, pq_num_centroids=128` |
| Faiss IVFPQ       | 0.864     | 0.220           | `nlist=1024, nprobe=10, m=32, nbits=7`                                                   |
| CompressedFastIVF | 0.933     | 0.050           | `nlist=1024, nprobe=10, ratio_threshold=0.5, compression_ndim=128`                       |
| CompressedFastIVF | 0.889     | 0.040           | `nlist=1024, nprobe=10, ratio_threshold=0.5, compression_ndim=64`                        |




## Custom mini batch k-means implementation 

Efficient mini-batch kmeans implementations with numba and numpy

```python
from fast_ivf.kmeans import MiniBatchKMeans
import numpy as np

kmeans = MiniBatchKMeans(num_centroids=16, batch_size=32, metric="l2")
data = np.random.rand(5000, 64)
kmeans.train(data)
kmeans.add(data)
labels = kmeans.predict(data)
```

Efficient mini-batch kmeans implementations to train product quantization centroids

```python
from fast_ivf.kmeans import SubvectorsMiniBatchKMeans
import numpy as np

kmeans = SubvectorsMiniBatchKMeans(num_centroids=16, num_subvectors=8, batch_size=32, metric="l2")
data = np.random.rand(5000, 64)
kmeans.train(data)
kmeans.add(data)
labels = kmeans.predict(data)
```

## FastIVF

Similar to `faiss.IndexIVFFlat( faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)`


```python
from fast_ivf import FastIVF
from fast_ivf.core import normalize
import numpy as np

nlist = 1024
train_embeddings = normalize(np.random.rand(10000, 512).astype(np.float32))
index = FastIVF(512, nlist=nlist)
index.train(train_embeddings)

index.nprobe = 10
# greedy skip voronoi cells which are having score smaller than 0.5 of the largest score
# higher values lead to faster search but less accurate
index.ratio_threshold = 0.5

test_embeddings = normalize(np.random.rand(100, 512).astype(np.float32))
distances, indices = index.search(test_embeddings, k=100)

```

## FastIVFPQ

Similar to `faiss_index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, m, n_bits)`

```python
from fast_ivf import FastIVFPQ

nlist = 1024
# pq_num_centroids = 2 ** n_bits
# pq_num_subvectors = m
index = FastIVFPQ(512, nlist=nlist, pq_num_centroids=64, pq_num_subvectors=32)
index.train(train_embeddings)
index.nprobe = 10
index.ratio_threshold = 0.5
distances, indices = index.search(test_embeddings, k=100)

# compute exact scores for top 100 results, this is slower but more accurate
distances, indices = index.search(test_embeddings, k=100, rescore=True)

# calibrate scores by fitting a linear regression model to N=20 exact scores, if -1 then all scores are exactly computed
index.rescore_num_samples = 20
distances, indices = index.search(test_embeddings, k=100, rescore=True)

```

## CompressedFastIVF

Trains keras autoencoder to compress embeddings to lower dimensionality


```python
from fast_ivf import CompressedFastIVF

nlist = 1024
index = CompressedFastIVF(512, nlist=nlist, compression_ndim=128)
index.train(train_embeddings)
index.nprobe = 10
index.ratio_threshold = 0.5
distances, indices = index.search(test_embeddings, k=100)

# compute exact scores for top 100 results, this is slower but more accurate
distances, indices = index.search(test_embeddings, k=100, rescore=True)

```



