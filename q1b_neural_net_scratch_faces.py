"""Template for a 3 layer feed forward neural network for binary face
classification, implemented from scratch.

Architecture: input -> hidden1 -> hidden2 -> output.

Implement the forward pass, back propagation, and weight update
yourself. You may use numpy. You may not use torch, tensorflow,
sklearn, jax, or keras for training, gradients, or prediction.

Required public API (fixed for auto grading):
  * class `ScratchNeuralNetworkFaces` with methods `forward`,
    `backward`, `update_weights`, `train`, `predict`, `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1b_neural_net_scratch_faces.py <training_percent>
"""

import sys
import time
import numpy as np

from util_faces import load_faces, flatten_images


class ScratchNeuralNetworkFaces:
    """3 layer fully connected network for binary face detection:

    input (70*60 = 4200) -> hidden1 -> hidden2 -> output (1 or 2).

    You may model the output as a single sigmoid unit (binary cross
    entropy) or as two softmax units. Either is fine; just be
    consistent in `forward`, `backward`, and `predict`.
    """

    def __init__(
        self,
        input_size: int = 70 * 60,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        output_size: int = 2,
        learning_rate: float = 0.01,
        num_epochs: int = 20,
        batch_size: int = 32,
        seed: int | None = None,
    ):
        """Initialise network hyperparameters and weight matrices."""
        # TODO: initialize weights, biases, and hyperparameters.
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_size), size=(input_size, hidden1_size))
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden1_size), size=(hidden1_size, hidden2_size))
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = rng.normal(0.0, np.sqrt(2.0 / hidden2_size), size=(hidden2_size, output_size))
        self.b3 = np.zeros((1, output_size))
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. See `ScratchNeuralNetworkDigits.forward`."""
        # TODO: 3 linear layers with hidden activations; final softmax
        # (or sigmoid if output_size == 1). Cache intermediates.
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, z2)
        z3 = a2 @ self.W3 + self.b3

        z3_shift = z3 - np.max(z3, axis=1, keepdims=True)
        exp_scores = np.exp(z3_shift)
        y_hat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "y_hat": y_hat}
        return y_hat

    def backward(self, X: np.ndarray, y_onehot: np.ndarray) -> dict:
        """Back propagate loss gradients through the network."""
        # TODO: compute dW1, db1, dW2, db2, dW3, db3 via the chain rule.
        m = X.shape[0]
        y_hat = self.cache["y_hat"]
        a2 = self.cache["a2"]
        z2 = self.cache["z2"]
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]

        dz3 = (y_hat - y_onehot) / m
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def update_weights(self, grads: dict) -> None:
        """Apply one gradient descent step using `grads` from `backward`."""
        # TODO: SGD update with self.learning_rate.
        self.W1 -= self.learning_rate * grads["dW1"]
        self.b1 -= self.learning_rate * grads["db1"]
        self.W2 -= self.learning_rate * grads["dW2"]
        self.b2 -= self.learning_rate * grads["db2"]
        self.W3 -= self.learning_rate * grads["dW3"]
        self.b3 -= self.learning_rate * grads["db3"]

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Full training loop: epochs and mini batches."""
        # TODO: flatten, one hot encode labels, epoch and mini batch loop.
        X = flatten_images(training_images)
        y = training_labels.astype(int)
        n = X.shape[0]

        for _ in range(self.num_epochs):
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                y_onehot = np.zeros((yb.shape[0], self.output_size))
                y_onehot[np.arange(yb.shape[0]), yb] = 1.0

                self.forward(xb)
                grads = self.backward(xb, y_onehot)
                self.update_weights(grads)

    def predict(self, image: np.ndarray) -> int:
        """Predict 0 or 1 for a single 70x60 image."""
        # TODO: flatten, run forward, argmax (or threshold).
        x = image.reshape(1, -1)
        probs = self.forward(x)
        return int(np.argmax(probs, axis=1)[0])

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        # TODO: vectorised forward pass, argmax, compare.
        X = flatten_images(images)
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the scratch NN on faces."""
    training_images, training_labels = load_faces("train")
    test_images, test_labels = load_faces("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        net = ScratchNeuralNetworkFaces()
        start = time.time()
        net.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = net.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== Scratch NN | Faces | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
