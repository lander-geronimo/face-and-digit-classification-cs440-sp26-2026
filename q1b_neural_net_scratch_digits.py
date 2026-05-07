"""Template for a 3 layer feed forward neural network for digit
classification, implemented from scratch.

Architecture: input -> hidden1 -> hidden2 -> output.

Implement the forward pass, back propagation, and weight update
yourself. You may use numpy for linear algebra. You may not use torch,
tensorflow, sklearn, jax, or keras for training, gradients, or
prediction.

Required public API (fixed for auto grading):
  * class `ScratchNeuralNetworkDigits` with methods `forward`,
    `backward`, `update_weights`, `train`, `predict`, `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1b_neural_net_scratch_digits.py <training_percent>
"""

import sys
import time
import numpy as np

from util_digits import load_digits, flatten_images


class ScratchNeuralNetworkDigits:
    """3 layer fully connected network: 784 to h1 to h2 to 10.

    Use any reasonable hidden activation (ReLU, sigmoid, tanh). For the
    output layer, softmax paired with cross entropy loss is typical.
    Document your choices in the report.

    Implementation notes:
      * Store weights as numpy arrays: W1 (784, h1), W2 (h1, h2),
        W3 (h2, 10), plus biases b1, b2, b3.
      * Initialise with small random values (scaled Gaussian, He,
        Xavier) to break symmetry.
      * `forward` should cache intermediate activations so that
        `backward` can compute gradients without re running forward.
    """

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        output_size: int = 10,
        learning_rate: float = 0.01,
        num_epochs: int = 20,
        batch_size: int = 32,
        seed: int | None = None,
    ):
        """Initialise network hyperparameters and weight matrices."""
        # TODO: initialize self.W1, self.b1, self.W2, self.b2, self.W3,
        # self.b3 and store the hyperparameters.
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
        """Forward pass.

        `X` has shape (N, 784). Return shape is (N, 10). You may return
        probabilities (after softmax) or raw logits; keep `predict` and
        `backward` consistent with your choice.
        """
        # TODO:
        # z1 = X W1 + b1 ; a1 = activation(z1)
        # z2 = a1 W2 + b2 ; a2 = activation(z2)
        # z3 = a2 W3 + b3 ; y_hat = softmax(z3)
        # Cache intermediates (e.g. self.cache) for the backward pass.
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
        """Back propagate loss gradients through the network.

        `X` has shape (N, 784); `y_onehot` has shape (N, 10). Return a
        dict like
        `{"dW1": ..., "db1": ..., "dW2": ..., "db2": ..., "dW3": ..., "db3": ...}`.
        """
        # TODO: compute gradients of the loss w.r.t. each weight and bias.
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
        # TODO: self.W1 -= self.learning_rate * grads["dW1"]  (etc.)
        self.W1 -= self.learning_rate * grads["dW1"]
        self.b1 -= self.learning_rate * grads["db1"]
        self.W2 -= self.learning_rate * grads["dW2"]
        self.b2 -= self.learning_rate * grads["db2"]
        self.W3 -= self.learning_rate * grads["dW3"]
        self.b3 -= self.learning_rate * grads["db3"]

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Full training loop: epochs and mini batches.

        `training_images` has shape (N, 28, 28). `training_labels` has
        shape (N,) with values in {0..9}.
        """
        # TODO: flatten images, one hot encode labels, loop over epochs
        # and mini batches, and call forward -> backward -> update_weights.
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
        """Predict a single label in {0..9} for a 28x28 image."""
        # TODO: flatten, run forward, argmax over the output layer.
        x = image.reshape(1, -1)
        probs = self.forward(x)
        return int(np.argmax(probs, axis=1)[0])

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        # TODO: vectorised: flatten all, forward once, argmax, compare.
        X = flatten_images(images)
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the scratch NN on digits."""
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        net = ScratchNeuralNetworkDigits()
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

    print(f"\n=== Scratch NN | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
