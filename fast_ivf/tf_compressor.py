from typing import Optional

import tensorflow as tf
from tqdm import tqdm
import numpy as np


class TFCentroidCompressor:
    def __init__(
        self,
        ndim: int,
        num_centroids: int,
        bottleneck: int = 32,
        learning_rate: float = 0.005,
        batch_size: int = 128,
        steps_per_epoch: int = 2000,
        epochs: int = 5,
        dropout: float = 0.2,
        label_smoothing: float = 0.01,
        activation: str = "relu",
        predict_approx_scores: bool = False,
    ):
        self.ndim = ndim
        self.bottleneck = bottleneck
        self.num_centroids = num_centroids
        self.batch_size = batch_size
        self.dropout = dropout
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.activation = activation
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.predict_approx_scores = predict_approx_scores

        self.model = self.build_model()
        self.projector_fn = None
        self._vectors: Optional[np.ndarray] = None
        self._history = None

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(self.bottleneck, name="projection"),
                tf.keras.layers.Activation(self.activation),
                tf.keras.layers.Dense(self.num_centroids, activation="softmax"),
            ]
        )
        return model

    def get_compressor(self):
        if self.projector_fn is None:
            if self.predict_approx_scores:
                outputs = [self.model.get_layer("projection").output, self.model.output]
            else:
                outputs = self.model.get_layer("projection").output
            projector = tf.keras.Model(inputs=self.model.input, outputs=outputs)
            projector = tf.function(projector)
            self.projector_fn = projector
        return self.projector_fn

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.label_smoothing
            ),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.TopKCategoricalAccuracy(k=10),
            ],
        )

    def train(self, iterator_fn, verbose: int = 1):
        print(
            f"Training compressor steps_per_epoch={self.steps_per_epoch}"
            f" epoch={self.epochs} compression_ndim={self.bottleneck}"
        )
        dataset = dataset_from_centroid_iterator(
            iterator_fn,
            num_centroids=self.num_centroids,
            ndim=self.ndim,
            batch_size=self.batch_size,
            shuffle_size=16 * self.batch_size,
        )

        self.compile()
        self.projector_fn = None
        history = self.model.fit(
            dataset,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            verbose=verbose,
        )
        self._history = history.history
        return history

    def predict(self, vectors: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        compressor = self.get_compressor()
        if self.predict_approx_scores:
            x_proj, x_centroids = compressor(vectors)
            x_centroids = x_centroids.numpy()
        else:
            x_proj = compressor(vectors)
            x_centroids = None
        x_proj = x_proj.numpy()
        x_proj = x_proj / np.linalg.norm(x_proj, axis=1, keepdims=True)
        return x_proj, x_centroids

    def add(self, iterator_fn, batch_size: int = 1024):
        iterator = iterator_fn(batch_size)
        x_compressed = []
        for x, y in tqdm(iterator, f"Adding compressed vectors"):
            x_proj = self.predict(x)[0]
            x_compressed.append(x_proj)

        x_compressed = np.concatenate(x_compressed)
        self._vectors = x_compressed


def dataset_from_centroid_iterator(
    iterator_fn,
    *,
    num_centroids: int,
    ndim: int,
    batch_size: int = 256,
    shuffle_size: int = 8000,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(
        iterator_fn,
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, ndim), (None,)),
    )

    def map_fn(x, y):
        y = tf.one_hot(y, num_centroids)
        return x, y

    dataset = dataset.map(map_fn)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
    dataset = dataset.repeat(-1)
    dataset = dataset.shuffle(shuffle_size).batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
