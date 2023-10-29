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



