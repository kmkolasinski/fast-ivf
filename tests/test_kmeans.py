import unittest

import numpy as np

from fast_ivf.kmeans import (
    SubvectorsMiniBatchKMeans,
    to_train_generator,
    MiniBatchKMeans,
    normalize,
    batch_compute_assignments,
    compute_assignments,
    batch_l2_distance_matrix,
    l2_distance_matrix,
    batch_update_assignments_add,
    batch_update_assignments_sub,
)


class TestMiniBatchKMeans(unittest.TestCase):
    def test_train(self):
        kmeans = MiniBatchKMeans(16, 32, metric="l2")
        data = np.random.rand(5000, 64)
        kmeans.train(data)
        kmeans.add(data)
        self.assertEqual(kmeans._centroids.shape, (64, 16))
        self.assertEqual(kmeans.num_vectors, 5000)

        labels = kmeans.predict(data)
        np.testing.assert_equal(kmeans._labels, labels)

        kmeans = MiniBatchKMeans(16, 32, metric="cosine")
        kmeans.train(data)
        kmeans.add(data)
        labels = kmeans.predict(data)
        np.testing.assert_equal(kmeans._labels, labels)


class TestSubspaceMiniBatchKMeans(unittest.TestCase):
    def test_init_centroids(self):
        kmeans = SubvectorsMiniBatchKMeans(16, 4, 512, metric="l2")

        data = np.random.rand(1000, 32)
        data = to_train_generator(data, kmeans.batch_size)

        centroids_sums, counts = kmeans.init_centroids(data)
        self.assertEqual(centroids_sums.shape, (4, 16, 8))
        self.assertEqual(counts.shape, (4, 16))

        centroids = kmeans.to_average_centroids(centroids_sums, counts)
        self.assertEqual(centroids.shape, (4, 8, 16))
        np.testing.assert_allclose(centroids[0].T, centroids_sums[0, :, :])
        np.testing.assert_allclose(centroids[1].T, centroids_sums[1, :, :])

        kmeans = SubvectorsMiniBatchKMeans(16, 4, 512, metric="cosine")
        centroids_sums, counts = kmeans.init_centroids(data)
        self.assertEqual(centroids_sums.shape, (4, 16, 8))
        self.assertEqual(counts.shape, (4, 16))

        centroids = kmeans.to_average_centroids(centroids_sums, counts)
        self.assertEqual(centroids.shape, (4, 8, 16))
        np.testing.assert_allclose(centroids[0].T, normalize(centroids_sums[0, :, :]))
        np.testing.assert_allclose(centroids[1].T, normalize(centroids_sums[1, :, :]))

    def test_batch_compute_assignments(self):
        kmeans = SubvectorsMiniBatchKMeans(16, 4, 512, metric="cosine")

        data = np.random.rand(1000, 32)
        data = to_train_generator(data, kmeans.batch_size)

        centroids_sums, counts = kmeans.init_centroids(data)
        centroids = kmeans.to_average_centroids(centroids_sums, counts)

        batch = next(data)
        batch = np.reshape(batch, (batch.shape[0], kmeans.num_subvectors, -1))
        batch = np.transpose(batch, (1, 0, 2))

        scores, indices = batch_compute_assignments(centroids, batch, "cosine")
        self.assertEqual(scores.shape, (4, 512, 16))
        self.assertEqual(indices.shape, (4, 512))

        scores_0, indices_0 = compute_assignments(centroids[0], batch[0], "cosine")

        np.testing.assert_allclose(scores_0, scores[0])
        np.testing.assert_equal(indices_0, indices[0])

    def test_batch_compute_assignments_l2(self):
        kmeans = SubvectorsMiniBatchKMeans(16, 4, 512, metric="l2")

        data = np.random.rand(1000, 32)
        data = to_train_generator(data, kmeans.batch_size)

        centroids_sums, counts = kmeans.init_centroids(data)
        centroids = kmeans.to_average_centroids(centroids_sums, counts)

        batch = next(data)
        batch = np.reshape(batch, (batch.shape[0], kmeans.num_subvectors, -1))
        batch = np.transpose(batch, (1, 0, 2))

        scores, indices = batch_compute_assignments(centroids, batch, "l2")
        self.assertEqual(scores.shape, (4, 512, 16))
        self.assertEqual(indices.shape, (4, 512))

        scores_0, indices_0 = compute_assignments(centroids[0], batch[0], "l2")

        np.testing.assert_allclose(scores_0, scores[0], atol=1e-5)
        np.testing.assert_equal(indices_0, indices[0])

    def test_batch_l2_distance_matrix(self):
        kmeans = SubvectorsMiniBatchKMeans(16, 4, 512, metric="l2")

        data = np.random.rand(1000, 32)
        data = to_train_generator(data, kmeans.batch_size)
        batch = next(data)
        batch = np.reshape(batch, (batch.shape[0], kmeans.num_subvectors, -1))
        batch = np.transpose(batch, (1, 0, 2))

        centroids_sums, counts = kmeans.init_centroids(data)
        centroids = kmeans.to_average_centroids(centroids_sums, counts)

        distances = batch_l2_distance_matrix(centroids, batch)
        distances_0 = l2_distance_matrix(centroids[0], batch[0])
        np.testing.assert_allclose(distances_0, distances[0], atol=1e-5)
        distances_1 = l2_distance_matrix(centroids[1], batch[1])
        np.testing.assert_allclose(distances_1, distances[1], atol=1e-5)

    def test_predict(self):
        kmeans = SubvectorsMiniBatchKMeans(16, 4, 512, metric="l2")

        data = np.random.rand(1000, 32)
        data = to_train_generator(data, kmeans.batch_size)

        centroids_sums, counts = kmeans.init_centroids(data)
        centroids = kmeans.to_average_centroids(centroids_sums, counts)
        kmeans._centroids = centroids

        batch = next(data)
        batch, (_, assignments) = kmeans.predict_all(batch)

        self.assertEqual(batch.shape, (4, 512, 8))
        self.assertEqual(assignments.shape, (4, 512))
        self.assertEqual(centroids_sums.shape, (4, 16, 8))
        self.assertEqual(counts.shape, (4, 16))

        batch_update_assignments_add(centroids_sums, counts, batch, assignments)
        batch_update_assignments_sub(centroids_sums, counts, batch, assignments)
        np.testing.assert_equal(counts, np.ones_like(counts))

    def test_train(self):
        kmeans = SubvectorsMiniBatchKMeans(16, 4, 512, metric="l2")
        data = np.random.rand(5000, 64)
        kmeans.train(data)
        kmeans.add(data)
        self.assertEqual(kmeans._labels.shape, (5000, 4))
        self.assertEqual(kmeans._vectors.shape, (5000, 64))

        _, (distances, labels) = kmeans.predict_all(data)
        np.testing.assert_equal(kmeans._labels, labels.T)

        test_data = np.random.rand(3, 64)
        test_distances = kmeans.predict_approximated_scores(
            test_data, np.array([0, 1, 2])
        )

        train_data = data[:3]
        train_distances = kmeans.predict_approximated_scores(
            train_data, np.array([0, 1, 2])
        )

        for i in range(3):
            self.assertLess(train_distances[i, i], test_distances[i, i])
