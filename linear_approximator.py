"""
Shared linear function approximator utilities.

This module provides a single reusable `LinearFunctionApproximator`
that converts states to features and implements `predict`/`update`.
It is intended to be reused by Monte Carlo, SARSA and Q-Learning
linear approximator implementations.
"""

import numpy as np


class LinearFunctionApproximator:
    """
    Linear function approximator for value / Q-function approximation.

    Supports: predict(state) -> scalar, and update(state, target)
    where the update uses gradient descent on squared error.
    """

    def __init__(self, num_features, learning_rate=0.01):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_features)

    def get_features(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        elif isinstance(state, (int, float)):
            state = np.array([state])

        # Normalize state values to [-1, 1] range (simple heuristic)
        state_normalized = np.clip(state / 100.0, -1, 1)

        features = np.zeros(self.num_features)
        features[0] = 1.0  # bias

        if self.num_features > 1 and len(state_normalized) > 0:
            features[1] = state_normalized[0]

        if self.num_features > 2 and len(state_normalized) > 0:
            features[2] = state_normalized[0] ** 2

        for i in range(3, min(self.num_features, 8)):
            center = (i - 3) * 0.2 - 0.5
            if len(state_normalized) > 0:
                features[i] = np.exp(-((state_normalized[0] - center) ** 2) / 0.1)

        return features

    def predict(self, state):
        features = self.get_features(state)
        return float(np.dot(self.weights, features))

    def update(self, state, target_value):
        features = self.get_features(state)
        prediction = np.dot(self.weights, features)
        error = target_value - prediction
        self.weights += self.learning_rate * error * features
