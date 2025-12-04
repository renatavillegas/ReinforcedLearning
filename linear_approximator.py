import numpy as np

class LinearFunctionApproximator:
    """
    Linear function approximator for value / Q-function approximation.
    Supports: predict(state) -> scalar, and update(state, target)
    where the update uses gradient descent on squared error.
    """
    def __init__(self, num_features, learning_rate=0.01, scale=100.0):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.scale = float(scale)
        self.weights = np.zeros(num_features)
    def get_features(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        elif isinstance(state, (int, float)):
            state = np.array([state])
        # Convert to float array and apply simple scaling to keep magnitudes reasonable
        state = np.asarray(state, dtype=float)
        if state.size == 0:
            state_normalized = state
        else:
            # normalize by provided scale (typically the initial price)
            state_normalized = state / self.scale
        # Linear-only features: bias + (at most) one coefficient per state dimension
        features = np.zeros(self.num_features, dtype=float)
        if self.num_features > 1 and state_normalized.size > 0:
            # number of linear features we can place
            n_lin = min(self.num_features - 1, state_normalized.size)
            features[1 : 1 + n_lin] = state_normalized[:n_lin]
        return features
    def predict(self, state):
        features = self.get_features(state)
        return float(np.dot(self.weights, features))
    def update(self, state, target_value):
        features = self.get_features(state)
        prediction = np.dot(self.weights, features)
        error = target_value - prediction
        self.weights += self.learning_rate * error * features