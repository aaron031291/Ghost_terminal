import logging
import json
import os
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
import uuid
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CaptureOutput:
    """Context manager to capture stdout and stderr."""
    def __init__(self):
        self.stdout_buffer = StringIO()
        self.stderr_buffer = StringIO()
        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
    
    def __enter__(self):
        sys.stdout = self.stdout_buffer
        sys.stderr = self.stderr_buffer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout_backup
        sys.stderr = self.stderr_backup
    
    def get_output(self) -> Tuple[str, str]:
        """Get captured stdout and stderr."""
        return self.stdout_buffer.getvalue(), self.stderr_buffer.getvalue()

@dataclass
class SandboxConfig:
    """Configuration for a sandbox environment."""
    name: str
    description: str
    allowed_modules: List[str]
    memory_limit_mb: int = 512
    execution_timeout_seconds: int = 30
    max_iterations: int = 100
    save_history: bool = True
    visualization_enabled: bool = True
    auto_feedback: bool = True

@dataclass
class SandboxResult:
    """Result of a sandbox execution."""
    success: bool
    output: str
    error: str = ""
    execution_time_ms: float = 0
    memory_used_mb: float = 0
    iterations: int = 0
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class Scenario:
    """A learning scenario with challenges and evaluation criteria."""
    id: str
    title: str
    description: str
    difficulty: str
    category: str
    instructions: List[str]
    starter_code: str
    test_cases: List[Dict[str, Any]]
    evaluation_criteria: List[str]
    hints: List[str] = field(default_factory=list)
    solution_code: str = ""
    time_limit_minutes: int = 60
    resources: List[Dict[str, str]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass
class ScenarioAttempt:
    """Record of an attempt to solve a scenario."""
    scenario_id: str
    user_id: str
    code_submitted: str
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[SandboxResult] = None
    completed: bool = False
    time_spent_seconds: float = 0
    iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        if self.result:
            result["result"] = self.result.to_dict()
        return result

class MLSandbox:
    """
    A sandbox environment for experimenting with ML models and algorithms.
    Provides isolated execution, visualization, and feedback.
    """
    
    def __init__(self, config: SandboxConfig):
        """Initialize the sandbox with the given configuration."""
        self.config = config
        self.history = []
        self.scenarios = self._load_scenarios()
        self.user_attempts = {}
        self.current_execution = None
        
    def _load_scenarios(self) -> Dict[str, Scenario]:
        """Load predefined learning scenarios."""
        scenarios = {}
        
        # Decision Tree Scenario
        scenarios["decision_tree_basic"] = Scenario(
            id="decision_tree_basic",
            title="Building a Basic Decision Tree Classifier",
            description="Implement a decision tree classifier from scratch to understand the "
                       "fundamentals of tree-based models.",
            difficulty="intermediate",
            category="tree_based_models",
            instructions=[
                "Implement a basic decision tree classifier with the following components:",
                "1. A function to calculate entropy or Gini impurity",
                "2. A function to find the best feature to split on",
                "3. A recursive function to build the tree",
                "4. A prediction function to classify new instances"
            ],
            starter_code="""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Predicted class (for leaf nodes)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X, y):
        # TODO: Implement the training algorithm
        self.root = self._grow_tree(X, y, depth=0)
        
    def _grow_tree(self, X, y, depth):
        # TODO: Implement the recursive tree-building algorithm
        pass
        
    def _best_split(self, X, y):
        # TODO: Find the best feature and threshold to split the data
        pass
        
    def _calculate_impurity(self, y):
        # TODO: Calculate entropy or Gini impurity
        pass
        
    def predict(self, X):
        # TODO: Implement the prediction function
        pass
        
    def _predict_sample(self, x, node):
        # TODO: Recursively traverse the tree to predict a single sample
        pass

# Example usage:
# X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# y_train = np.array([0, 0, 0, 1])
# clf = DecisionTreeClassifier(max_depth=2)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_train)
""",
            test_cases=[
                {
                    "input": {
                        "X_train": [[0, 0], [1, 0], [0, 1], [1, 1]],
                        "y_train": [0, 0, 0, 1],
                        "X_test": [[0, 0], [1, 0], [0, 1], [1, 1]]
                    },
                    "expected_output": [0, 0, 0, 1],
                    "description": "Basic XOR-like pattern"
                },
                {
                    "input": {
                        "X_train": [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]],
                        "y_train": [0, 0, 0, 1, 1],
                        "X_test": [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]]
                    },
                    "expected_output": [0, 0, 0, 1],
                    "description": "Testing with points near decision boundaries"
                }
            ],
            evaluation_criteria=[
                "Correctness of implementation",
                "Handling of edge cases",
                "Code efficiency and readability",
                "Understanding of decision tree concepts"
            ],
            hints=[
                "Start by implementing the impurity calculation function",
                "For the best split, iterate through all features and possible thresholds",
                "Remember to handle the base cases in the recursive tree-building function",
                "Consider using a depth-first approach for the prediction function"
            ],
            solution_code="""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class DecisionNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Predicted class (for leaf nodes)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._grow_tree(X, y, depth=0)
        
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples < 2:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # Split the data
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return DecisionNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
        
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        
        # If there's only one sample or all features are the same, return arbitrary values
        if n_samples <= 1 or np.all(X == X[0, :], axis=0).any():
            return 0, X[0, 0]
        
        best_gain = -1
        best_feature = 0
        best_threshold = 0
        
        current_impurity = self._calculate_impurity(y)
        
        # For each feature
        for feature_idx in range(n_features):
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature_idx])
            
            # For each possible threshold
            for threshold in thresholds:
                # Split the data
                left_idxs = X[:, feature_idx] < threshold
                right_idxs = ~left_idxs
                
                # Skip if one of the splits is empty
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue
                
                # Calculate the weighted impurity
                left_impurity = self._calculate_impurity(y[left_idxs])
                right_impurity = self._calculate_impurity(y[right_idxs])
                
                # Calculate the weights (proportion of samples)
                left_weight = np.sum(left_idxs) / n_samples
                right_weight = np.sum(right_idxs) / n_samples
                
                # Calculate the information gain
                gain = current_impurity - (left_weight * left_impurity + right_weight * right_impurity)
                
                # Update the best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
        
    def _calculate_impurity(self, y):
        # Gini impurity
        m = len(y)
        if m == 0:
            return 0
        
        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        
        # Calculate Gini impurity
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _most_common_label(self, y):
        # Return the most common class label
        unique_labels, counts = np.unique(y, return_counts=True)
        return unique_labels[np.argmax(counts)]
        
    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_sample(sample, self.root) for sample in X])
        
    def _predict_sample(self, x, node):
        # If we're at a leaf node, return the predicted class
        if node.value is not None:
            return node.value
        
        # Otherwise, decide which subtree to traverse
        if x[node.feature_idx] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

# Example usage:
# X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# y_train = np.array([0, 0, 0, 1])
# clf = DecisionTreeClassifier(max_depth=2)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_train)
"""
        )
        
        # Clustering Scenario
        scenarios["kmeans_clustering"] = Scenario(
            id="kmeans_clustering",
            title="Implementing K-Means Clustering",
            description="Build a K-Means clustering algorithm from scratch to understand the "
                       "fundamentals of unsupervised learning.",
            difficulty="intermediate",
            category="clustering",
            instructions=[
                "Implement a K-Means clustering algorithm with the following components:",
                "1. A function to initialize cluster centroids",
                "2. A function to assign data points to the nearest centroid",
                "3. A function to update centroids based on assigned points",
                "4. A main loop that iterates until convergence or max iterations"
            ],
            starter_code="""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iterations=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol  # Convergence tolerance
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        # TODO: Implement the K-Means algorithm
                X = np.array(X)
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iterations):
            # TODO: Implement the main K-Means loop
            pass
        
        return self
        
    def _initialize_centroids(self, X):
        # TODO: Initialize the centroids (randomly or using K-Means++)
        pass
        
    def _assign_clusters(self, X):
        # TODO: Assign each data point to the nearest centroid
        pass
        
    def _update_centroids(self, X, labels):
        # TODO: Update centroids based on the mean of assigned points
        pass
        
    def predict(self, X):
        # TODO: Predict the cluster for new data points
        pass
        
    def plot_clusters(self, X):
        # Helper function to visualize the clusters (for 2D data)
        if X.shape[1] != 2:
            print("Plotting only works for 2D data")
            return
            
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis', alpha=0.7)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='X', s=200, alpha=1)
        plt.title('K-Means Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

# Example usage:
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# kmeans = KMeansClustering(n_clusters=2)
# kmeans.fit(X)
# kmeans.plot_clusters(X)
""",
            test_cases=[
                {
                    "input": {
                        "X": [[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]],
                        "n_clusters": 2
                    },
                    "validation": "cluster_validation",
                    "description": "Basic two-cluster separation"
                },
                {
                    "input": {
                        "X": [[0, 0], [1, 0], [0, 1], [10, 10], [10, 11], [11, 10], [5, 5], [6, 5], [5, 6]],
                        "n_clusters": 3
                    },
                    "validation": "cluster_validation",
                    "description": "Three distinct clusters"
                }
            ],
            evaluation_criteria=[
                "Correctness of implementation",
                "Handling of edge cases (empty clusters, convergence)",
                "Code efficiency and readability",
                "Understanding of K-Means concepts"
            ],
            hints=[
                "For initialization, you can randomly select K data points as initial centroids",
                "Use Euclidean distance to assign points to the nearest centroid",
                "When updating centroids, handle the case where a cluster might be empty",
                "Check for convergence by comparing the change in centroids between iterations"
            ],
            solution_code="""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iterations=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol  # Convergence tolerance
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        X = np.array(X)
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iterations):
            # Assign clusters
            self.labels = self._assign_clusters(X)
            
            # Store old centroids for convergence check
            old_centroids = np.copy(self.centroids)
            
            # Update centroids
            self.centroids = self._update_centroids(X, self.labels)
            
            # Check for convergence
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break
        
        return self
        
    def _initialize_centroids(self, X):
        # Randomly select K data points as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
        
    def _assign_clusters(self, X):
        # Calculate distance from each point to each centroid
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)
        
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            # Get points assigned to cluster k
            cluster_points = X[labels == k]
            
            # Handle empty clusters
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If a cluster is empty, reinitialize its centroid
                new_centroids[k] = X[np.random.choice(X.shape[0])]
                
        return new_centroids
        
    def predict(self, X):
        X = np.array(X)
        # Calculate distance from each point to each centroid
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)
        
    def plot_clusters(self, X):
        # Helper function to visualize the clusters (for 2D data)
        if X.shape[1] != 2:
            print("Plotting only works for 2D data")
            return
            
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis', alpha=0.7)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='X', s=200, alpha=1)
        plt.title('K-Means Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

# Example usage:
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# kmeans = KMeansClustering(n_clusters=2)
# kmeans.fit(X)
# kmeans.plot_clusters(X)
"""
        )
        
        # Neural Network Scenario
        scenarios["neural_network_basic"] = Scenario(
            id="neural_network_basic",
            title="Building a Simple Neural Network",
            description="Implement a basic neural network from scratch to understand the fundamentals "
                       "of forward and backward propagation.",
            difficulty="advanced",
            category="neural_networks",
            instructions=[
                "Implement a simple neural network with one hidden layer that can:",
                "1. Initialize weights and biases properly",
                "2. Perform forward propagation",
                "3. Calculate loss",
                "4. Perform backward propagation to update weights",
                "5. Train on a dataset and make predictions"
            ],
            starter_code="""
import numpy as np
from typing import List, Tuple, Dict, Any

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        # TODO: Initialize weights and biases with proper scaling
        
    def sigmoid(self, x):
        # TODO: Implement sigmoid activation function
        pass
        
    def sigmoid_derivative(self, x):
        # TODO: Implement derivative of sigmoid function
        pass
        
    def forward(self, X):
        # TODO: Implement forward propagation
        pass
        
    def compute_loss(self, y_true, y_pred):
        # TODO: Implement mean squared error loss
        pass
        
    def backward(self, X, y):
        # TODO: Implement backward propagation
        pass
        
    def train(self, X, y, epochs=1000):
        # TODO: Implement training loop
        pass
        
    def predict(self, X):
        # TODO: Implement prediction function
        pass
        
    def evaluate(self, X, y):
        # TODO: Implement evaluation function
        pass

# Example usage:
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])  # XOR problem
# nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
# nn.train(X, y, epochs=10000)
# predictions = nn.predict(X)
""",
            test_cases=[
                {
                    "input": {
                        "X_train": [[0, 0], [0, 1], [1, 0], [1, 1]],
                        "y_train": [[0], [1], [1], [0]],
                        "X_test": [[0, 0], [0, 1], [1, 0], [1, 1]],
                        "hidden_size": 4,
                        "epochs": 5000
                    },
                    "validation": "nn_validation",
                    "description": "XOR problem - tests basic neural network capabilities"
                },
                {
                    "input": {
                        "X_train": [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
                                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                        "y_train": [[0], [1], [1], [0], [1], [0], [0], [1]],
                        "X_test": [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
                                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                        "hidden_size": 6,
                        "epochs": 8000
                    },
                    "validation": "nn_validation",
                    "description": "3-input parity problem - tests scalability"
                }
            ],
            evaluation_criteria=[
                "Correctness of implementation",
                "Proper weight initialization",
                "Correct gradient calculation",
                "Training convergence",
                "Code efficiency and readability"
            ],
            hints=[
                "Initialize weights with small random values to break symmetry",
                "Be careful with matrix dimensions in forward and backward propagation",
                "Use vectorized operations for efficiency",
                "The XOR problem requires at least one hidden layer",
                "Consider adding regularization if you're experiencing overfitting"
            ],
            solution_code="""
import numpy as np
from typing import List, Tuple, Dict, Any

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases with proper scaling
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
        
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
        
    def forward(self, X):
        # Forward propagation
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
        
    def compute_loss(self, y_true, y_pred):
        # Mean squared error loss
        m = y_true.shape[0]
        return np.sum((y_true - y_pred) ** 2) / (2 * m)
        
    def backward(self, X, y):
        m = X.shape[0]
        
        # Backward propagation
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
    def train(self, X, y, epochs=1000):
        X = np.array(X)
        y = np.array(y)
        
        losses = []
        for i in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            # Print loss every 1000 epochs
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {loss}")
                
        return losses
        
    def predict(self, X):
        X = np.array(X)
        # Forward pass
        y_pred = self.forward(X)
        # Convert probabilities to binary predictions
        return (y_pred > 0.5).astype(int)
        
        def evaluate(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y)
        
        # Calculate loss
        loss = self.compute_loss(y, self.forward(X))
        
        return {
            'accuracy': accuracy,
            'loss': loss
        }

# Example usage:
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])  # XOR problem
# nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
# nn.train(X, y, epochs=10000)
# predictions = nn.predict(X)
"""
        )
        
        # Real-world ML Scenario
        scenarios["recommendation_system"] = Scenario(
            id="recommendation_system",
            title="Building a Simple Recommendation System",
            description="Implement a content-based recommendation system for suggesting similar items.",
            difficulty="advanced",
            category="applied_ml",
            instructions=[
                "Build a content-based recommendation system that can:",
                "1. Process item features and convert them to numerical vectors",
                "2. Calculate similarity between items",
                "3. Recommend similar items based on a query item",
                "4. Evaluate the quality of recommendations"
            ],
            starter_code="""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.item_ids = None
        self.feature_matrix = None
        
    def fit(self, items_df, feature_columns, item_id_column='item_id'):
        # TODO: Process item features and create feature vectors
        pass
        
    def _preprocess_features(self, items_df, feature_columns):
        # TODO: Convert features to a format suitable for vectorization
        pass
        
    def _vectorize_features(self, feature_text):
        # TODO: Convert text features to TF-IDF vectors
        pass
        
    def get_similar_items(self, item_id, n=5):
        # TODO: Find and return n most similar items
        pass
        
    def recommend_for_item(self, item_id, n=5):
        # TODO: Generate recommendations based on item similarity
        pass
        
    def evaluate(self, test_data):
        # TODO: Evaluate recommendation quality
        pass

# Example usage:
# items_df = pd.DataFrame({
#     'item_id': [1, 2, 3, 4, 5],
#     'title': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'],
#     'category': ['electronics', 'electronics', 'books', 'books', 'clothing'],
#     'description': ['A great electronic device', 'Another electronic item', 
#                    'An interesting book', 'A classic novel', 'Stylish clothing']
# })
# recommender = ContentBasedRecommender()
# recommender.fit(items_df, feature_columns=['category', 'description'])
# similar_items = recommender.recommend_for_item(1, n=2)
""",
            test_cases=[
                {
                    "input": {
                        "items_df": {
                            "item_id": [1, 2, 3, 4, 5],
                            "title": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"],
                            "category": ["electronics", "electronics", "books", "books", "clothing"],
                            "description": ["A great electronic device", "Another electronic item", 
                                           "An interesting book", "A classic novel", "Stylish clothing"]
                        },
                        "feature_columns": ["category", "description"],
                        "query_item": 1,
                        "n": 2
                    },
                    "expected_output": [2, 3],  # Item 2 (same category) and Item 3 should be recommended
                    "description": "Basic recommendation test"
                },
                {
                    "input": {
                        "items_df": {
                            "item_id": [101, 102, 103, 104, 105, 106],
                            "title": ["Smartphone", "Laptop", "Python Book", "JavaScript Book", "T-shirt", "Jeans"],
                            "category": ["electronics", "electronics", "books", "books", "clothing", "clothing"],
                            "description": ["A modern smartphone with great camera", "Powerful laptop for developers", 
                                           "Learn Python programming", "JavaScript for web development", 
                                           "Cotton t-shirt", "Denim jeans"]
                        },
                        "feature_columns": ["category", "description"],
                        "query_item": 103,
                        "n": 3
                    },
                    "expected_output": [104, 102, 101],  # Books first, then electronics
                    "description": "More complex recommendation test"
                }
            ],
            evaluation_criteria=[
                "Correctness of implementation",
                "Quality of recommendations",
                "Handling of edge cases",
                "Code efficiency and readability"
            ],
            hints=[
                "Combine multiple features into a single text representation",
                "TF-IDF vectorization works well for text features",
                "Cosine similarity is a good metric for content-based recommendations",
                "Consider weighting different features based on their importance",
                "Handle the case where the query item doesn't exist"
            ],
            solution_code="""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.item_ids = None
        self.feature_matrix = None
        self.vectorizer = None
        
    def fit(self, items_df, feature_columns, item_id_column='item_id'):
        # Store item IDs
        self.item_ids = items_df[item_id_column].values
        
        # Process and combine features
        feature_text = self._preprocess_features(items_df, feature_columns)
        
        # Vectorize features
        self.feature_matrix, self.vectorizer = self._vectorize_features(feature_text)
        
        return self
        
    def _preprocess_features(self, items_df, feature_columns):
        # Combine all features into a single text representation
        feature_text = []
        
        for _, row in items_df.iterrows():
            text = ""
            for col in feature_columns:
                # Add column name as prefix to give more weight to categorical features
                if col in ['category', 'tags', 'genre']:
                    text += f"{col}_{row[col]} "
                else:
                    text += f"{row[col]} "
            feature_text.append(text.strip())
            
        return feature_text
        
    def _vectorize_features(self, feature_text):
        # Convert text features to TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
        feature_matrix = vectorizer.fit_transform(feature_text)
        
        return feature_matrix, vectorizer
        
    def get_similar_items(self, item_id, n=5):
        # Find the index of the item
        try:
            idx = np.where(self.item_ids == item_id)[0][0]
        except IndexError:
            return []
            
        # Calculate similarity with all items
        item_vector = self.feature_matrix[idx:idx+1]
        similarity_scores = cosine_similarity(item_vector, self.feature_matrix).flatten()
        
        # Get indices of most similar items (excluding the item itself)
        similar_indices = np.argsort(similarity_scores)[::-1]
        similar_indices = similar_indices[similar_indices != idx][:n]
        
        # Return item IDs
        return self.item_ids[similar_indices].tolist()
        
    def recommend_for_item(self, item_id, n=5):
        return self.get_similar_items(item_id, n)
        
    def evaluate(self, test_data):
        """
        Evaluate recommendation quality using precision@k
        
        test_data: List of tuples (item_id, relevant_items)
        where relevant_items is a list of item_ids that are known to be relevant
        """
        precisions = []
        recalls = []
        
        for item_id, relevant_items in test_data:
            recommended_items = self.recommend_for_item(item_id, n=len(relevant_items))
            
            # Calculate precision: how many recommended items are relevant
            if recommended_items:
                relevant_and_recommended = set(relevant_items) & set(recommended_items)
                precision = len(relevant_and_recommended) / len(recommended_items)
                recall = len(relevant_and_recommended) / len(relevant_items)
            else:
                precision = 0
                recall = 0
                
            precisions.append(precision)
            recalls.append(recall)
            
        return {
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls)
        }

# Example usage:
# items_df = pd.DataFrame({
#     'item_id': [1, 2, 3, 4, 5],
#     'title': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'],
#     'category': ['electronics', 'electronics', 'books', 'books', 'clothing'],
#     'description': ['A great electronic device', 'Another electronic item', 
#                    'An interesting book', 'A classic novel', 'Stylish clothing']
# })
# recommender = ContentBasedRecommender()
# recommender.fit(items_df, feature_columns=['category', 'description'])
# similar_items = recommender.recommend_for_item(1, n=2)
"""
        )
        
        # Anomaly Detection Scenario
        scenarios["anomaly_detection"] = Scenario(
            id="anomaly_detection",
            title="Implementing Anomaly Detection",
            description="Build an anomaly detection system using statistical methods and machine learning.",
            difficulty="advanced",
            category="applied_ml",
            instructions=[
                "Implement an anomaly detection system that can:",
                "1. Learn the normal distribution of features",
                "2. Identify anomalies using statistical methods",
                "3. Evaluate the performance of the anomaly detection",
                "4. Visualize the results"
            ],
            starter_code="""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_recall_curve, auc

class AnomalyDetector:
    def __init__(self, method='gaussian'):
        self.method = method
        self.parameters = {}
        
    def fit(self, X):
        # TODO: Learn the parameters of the normal distribution
        pass
        
    def _fit_gaussian(self, X):
        # TODO: Calculate mean and variance for each feature
        pass
        
    def compute_anomaly_score(self, X):
        # TODO: Calculate anomaly score for each instance
        pass
        
    def predict(self, X, threshold=None):
        # TODO: Predict which instances are anomalies
        pass
        
    def evaluate(self, X, y_true):
        # TODO: Evaluate the anomaly detection performance
        pass
        
    def visualize(self, X, predictions=None):
        # TODO: Visualize the data and anomalies (for 2D data)
        pass

# Example usage:
# # Generate normal data
# normal_data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=1000)
# # Generate anomalies
# anomalies = np.random.multivariate_normal(mean=[4, 4], cov=[[0.5, 0], [0, 0.5]], size=50)
# # Combine data
# X = np.vstack([normal_data, anomalies])
# y_true = np.zeros(X.shape[0])
# y_true[1000:] = 1  # Mark anomalies
#
# detector = AnomalyDetector()
# detector.fit(normal_data)  # Train only on normal data
# predictions = detector.predict(X)
# detector.visualize(X, predictions)
""",
            test_cases=[
                {
                    "input": {
                        "normal_mean": [0, 0],
                        "normal_cov": [[1, 0], [0, 1]],
                        "normal_size": 1000,
                        "anomaly_mean": [4, 4],
                        "anomaly_cov": [[0.5, 0], [0, 0.5]],
                        "anomaly_size": 50
                    },
                    "validation": "anomaly_validation",
                    "description": "Basic anomaly detection test with well-separated clusters"
                },
                {
                    "input": {
                        "normal_mean": [0, 0],
                        "normal_cov": [[1, 0.5], [0.5, 1]],
                        "normal_size": 1000,
                        "anomaly_mean": [2, 2],
                        "anomaly_cov": [[0.8, 0], [0, 0.8]],
                        "anomaly_size": 100
                    },
                    "validation": "anomaly_validation",
                    "description": "More challenging anomaly detection with overlapping distributions"
                }
            ],
            evaluation_criteria=[
                "Correctness of implementation",
                "Accuracy of anomaly detection",
                "Handling of edge cases",
                "Visualization quality",
                "Code efficiency and readability"
            ],
            hints=[
                "For Gaussian-based anomaly detection, calculate the probability density for each point",
                "Use a threshold on the probability density to identify anomalies",
                "Consider using precision-recall curves to evaluate performance",
                "For multivariate data, account for feature correlations",
                "Normalize features if they have different scales"
            ],
            solution_code="""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_recall_curve, auc, f1_score

class AnomalyDetector:
    def __init__(self, method='gaussian'):
        self.method = method
        self.parameters = {}
        self.threshold = None
        
    def fit(self, X):
        X = np.array(X)
        if self.method == 'gaussian':
            self._fit_gaussian(X)
        else:
            raise ValueError(f"Method {self.method} not supported")
        return self
        
    def _fit_gaussian(self, X):
        # Calculate mean and variance for each feature
        self.parameters['mu'] = np.mean(X, axis=0)
        self.parameters['sigma2'] = np.var(X, axis=0)
        
        # Handle features with zero variance
        self.parameters['sigma2'] = np.maximum(self.parameters['
        # Handle features with zero variance
        self.parameters['sigma2'] = np.maximum(self.parameters['sigma2'], 1e-8)
        
    def compute_anomaly_score(self, X):
        X = np.array(X)
        if self.method == 'gaussian':
            return self._compute_gaussian_score(X)
        else:
            raise ValueError(f"Method {self.method} not supported")
        
    def _compute_gaussian_score(self, X):
        # Calculate the probability density for each instance
        mu = self.parameters['mu']
        sigma2 = self.parameters['sigma2']
        
        # For each feature, calculate p(x)
        n_features = X.shape[1]
        p_x = np.ones((X.shape[0],))
        
        for j in range(n_features):
            # Calculate Gaussian probability density
            p_x_j = (1 / np.sqrt(2 * np.pi * sigma2[j])) * \
                    np.exp(-(X[:, j] - mu[j])**2 / (2 * sigma2[j]))
            p_x *= p_x_j
            
        # Return negative log probability as anomaly score (higher = more anomalous)
        return -np.log(p_x + 1e-10)  # Add small constant to avoid log(0)
        
    def predict(self, X, threshold=None):
        X = np.array(X)
        # Compute anomaly scores
        scores = self.compute_anomaly_score(X)
        
        # If threshold is not provided, use the one from cross-validation or a default
        if threshold is None:
            if self.threshold is None:
                # Default: mark top 5% as anomalies
                self.threshold = np.percentile(scores, 95)
            threshold = self.threshold
            
        # Predict anomalies (1 = anomaly, 0 = normal)
        predictions = (scores > threshold).astype(int)
        
        return predictions, scores
        
    def evaluate(self, X, y_true):
        X = np.array(X)
        y_true = np.array(y_true)
        
        # Compute anomaly scores
        _, scores = self.predict(X)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        
        # Find the best threshold based on F1 score
        f1_scores = []
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            f1 = f1_score(y_true, predictions)
            f1_scores.append(f1)
            
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # Set the best threshold
        self.threshold = best_threshold
        
        # Make final predictions with the best threshold
        predictions, _ = self.predict(X, threshold=best_threshold)
        
        return {
            'precision_recall_auc': pr_auc,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'predictions': predictions
        }
        
    def visualize(self, X, predictions=None, scores=None):
        X = np.array(X)
        
        # Only works for 2D data
        if X.shape[1] != 2:
            print("Visualization only works for 2D data")
            return
            
        plt.figure(figsize=(12, 10))
        
        # If predictions are not provided, compute them
        if predictions is None:
            predictions, scores = self.predict(X)
            
        # Plot normal points and anomalies
        normal_idx = np.where(predictions == 0)[0]
        anomaly_idx = np.where(predictions == 1)[0]
        
        plt.scatter(X[normal_idx, 0], X[normal_idx, 1], c='blue', label='Normal', alpha=0.6)
        plt.scatter(X[anomaly_idx, 0], X[anomaly_idx, 1], c='red', label='Anomaly', alpha=0.6)
        
        # If we have scores, plot contours
        if scores is not None:
            # Create a grid for contour plotting
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            # Compute anomaly scores for the grid
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            grid_scores = self.compute_anomaly_score(grid_points)
            grid_scores = grid_scores.reshape(xx.shape)
            
            # Plot contours
            contour = plt.contourf(xx, yy, grid_scores, levels=10, cmap='viridis', alpha=0.3)
            plt.colorbar(contour, label='Anomaly Score')
            
            # Plot decision boundary
            if self.threshold is not None:
                plt.contour(xx, yy, grid_scores, levels=[self.threshold], colors='k', linestyles='dashed')
        
        plt.title('Anomaly Detection Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

# Example usage:
# # Generate normal data
# normal_data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=1000)
# # Generate anomalies
# anomalies = np.random.multivariate_normal(mean=[4, 4], cov=[[0.5, 0], [0, 0.5]], size=50)
# # Combine data
# X = np.vstack([normal_data, anomalies])
# y_true = np.zeros(X.shape[0])
# y_true[1000:] = 1  # Mark anomalies
#
# detector = AnomalyDetector()
# detector.fit(normal_data)  # Train only on normal data
# predictions = detector.predict(X)
# detector.visualize(X, predictions)
"""
        )
        
        return scenarios
    
    def _create_learning_paths(self) -> Dict[str, List[str]]:
        """Create predefined learning paths with scenario sequences."""
        paths = {
            "ml_fundamentals": [
                "decision_tree_basic",
                "kmeans_clustering",
                "neural_network_basic"
            ],
            "applied_ml": [
                "decision_tree_basic",
                "recommendation_system",
                "anomaly_detection"
            ],
            "data_science": [
                "kmeans_clustering",
                "anomaly_detection",
                "recommendation_system"
            ]
        }
        return paths
    
    def execute_code(self, code: str, scenario_id: str = None) -> SandboxResult:
        """
        Execute user code in a sandbox environment with proper isolation.
        Returns execution results, output, and feedback.
        """
        start_time = time.time()
        
        # Create a safe execution environment
        local_vars = {}
        global_vars = {
            'np': np,
            'plt': plt,
            'pd': pd if 'pd' in sys.modules else None,
            'time': time,
            'random': random
        }
        
        # Capture output
        with CaptureOutput() as output:
            try:
                # Execute the code
                exec(code, global_vars, local_vars)
                success = True
                error_msg = ""
                
                # If scenario is provided, validate against test cases
                if scenario_id and scenario_id in self.scenarios:
                    scenario = self.scenarios[scenario_id]
                    feedback, success = self._validate_scenario(scenario, local_vars)
                else:
                    feedback = "Code executed successfully."
                    
            except Exception as e:
                success = False
                error_msg = f"{type(e).__name__}: {str(e)}"
                feedback = f"Error in code execution: {error_msg}"
        
        stdout, stderr = output.get_output()
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create and return result
        result = SandboxResult(
            success=success,
            output=stdout,
            error=stderr if stderr else error_msg,
            execution_time_ms=execution_time,
            memory_used_mb=0,  # Would need more complex monitoring to measure this accurately
            feedback=feedback
        )
        
        return result
    
    def _validate_scenario(self, scenario: Scenario, local_vars: Dict[str, Any]) -> Tuple[str, bool]:
        """Validate user code against scenario test cases."""
        feedback = []
        all_passed = True
        
        # Check if required classes or functions are defined
        required_objects = self._extract_required_objects(scenario.starter_code)
        for obj_name in required_objects:
            if obj_name not in local_vars:
                return f"Required object '{obj_name}' is not defined in your code.", False
        
        # Run test cases
        for i, test_case in enumerate(scenario.test_cases):
            test_result = self._run_test_case(test_case, local_vars)
            if test_result['passed']:
                feedback.append(f"✅ Test case {i+1} ({test_case['description']}): Passed")
            else:
                all_passed = False
                feedback.append(f"❌ Test case {i+1} ({test_case['description']}): Failed")
                feedback.append(f"   Reason: {test_result['message']}")
                
                # Add a hint if available
                if i < len(scenario.hints):
                    feedback.append(f"   Hint: {scenario.hints[i]}")
        
        # Add overall feedback
        if all_passed:
            feedback.append("\n🎉 Congratulations! All test cases passed.")
            feedback.append("Key concepts demonstrated in this scenario:")
            for criterion in scenario.evaluation_criteria:
                feedback.append(f"- {criterion}")
        else:
            feedback.append("\n⚠️ Some test cases failed. Review the feedback and try again.")
            
        return "\n".join(feedback), all_passed
    
    def _extract_required_objects(self, code: str) -> List[str]:
        """Extract required class or function names from starter code."""
        import re
        class_pattern = r"class\s+(\w+)"
        function_pattern = r"def\s+(\w+)"
        
        classes = re.findall(class_pattern, code)
        functions = re.findall(function_pattern, code)
        
        return classes + functions
    
    def _run_test_case(self, test_case: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case against user code."""
        try:
            if 'validation' in test_case:
                # Custom validation function
                validation_func = getattr(self, f"_{test_case['validation']}")
                return validation_func(test_case, local_vars)
            else:
                # Standard input/output validation
                input_data = test_case['input']
                expected_output = test_case['expected_output']
                
                # Execute user code with test input
                actual_output = self._execute_test(input_data, local_vars)
                
                # Compare results
                if isinstance(expected_output, list) and isinstance(actual_output, list):
                    if set(expected_output) == set(actual_output):
                        return {'passed': True, 'message': 'Test passed'}
                    else:
                        return {
                            'passed': False, 
                            'message': f'Expected output {expected_output}, but got {actual_output}'
                        }
                elif np.array_equal(np.array(expected_output), np.array(actual_output)):
                    return {'passed': True, 'message': 'Test passed'}
                else:
                    return {
                        'passed': False, 
                        'message': f'Expected output {expected_output}, but got {actual_output}'
                    }
        except Exception as e:
            return {'passed': False, 'message': f'Error during test execution: {str(e)}'}
    
    def _execute_test(self, input_data: Dict[str, Any], local_vars: Dict[str, Any]) -> Any:
        """Execute a test with the given input data."""
        # This is a simplified implementation - would need to be adapted for each scenario
        if isinstance(input_data, dict):
            # Example for decision tree scenario
            if all(k in input_data for k in ['X_train', 'y_train', 'X_test']):
                # Find the classifier class
                clf_class = None
                for var_name, var_value in local_vars.items():
                    if var_name.endswith('Classifier') and callable(getattr(var_value, 'fit', None)):
                        clf_class = var_value
                        break
                
                if clf_class is None:
                    raise ValueError("Could not find classifier class in your code")
                
                # Create instance with parameters if provided
                clf_params = {}
                if 'hidden_size' in input_data:
                    clf_params['hidden_size'] = input_data['hidden_size']
                if 'max_depth' in input_data:
                    clf_params['max_depth'] = input_data['max_depth']
                
                clf = clf_class(**clf_params)
                
                # Train and predict
                clf.fit(np.array(input_data['X_train']), np.array(input_data['y_train']))
                return clf.predict(np.array(input_data['X_test'])).tolist()
            
            # Example for recommendation system
            elif all(k in input_data for k in ['items_df', 'feature_columns', 'query_item']):
                # Find the recommender class
                recommender_class = None
                for var_name, var_value in local_vars.items():
                    if var_name.endswith('Recommender') and callable(getattr(var_value, 'fit', None)):
                        recommender_class = var_value
                        break
                
                if recommender_class is None:
                    raise ValueError("Could not find recommender class in your code")
                
                # Create dataframe
                items_df = pd.DataFrame(input_data['items_df'])
                
                # Create instance and train
                recommender = recommender_class()
                recommender.fit(items_df, input_data['feature_columns'])
                
                # Get recommendations
                return recommender.recommend_for_item(input_data['query_item'], input_data['n'])
        
        return None
    
    def _cluster_validation(self, test_case: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, bool]:
        """Validate K-means clustering implementation."""
        try:
            # Find the KMeans class
            kmeans_class = None
            for var_name, var_value in local_vars.items():
                                if var_name.endswith('Clustering') and callable(getattr(var_value, 'fit', None)):
                    kmeans_class = var_value
                    break
            
            if kmeans_class is None:
                return {'passed': False, 'message': "Could not find KMeans clustering class in your code"}
            
            # Get test data
            input_data = test_case['input']
            X = np.array(input_data['X'])
            n_clusters = input_data['n_clusters']
            
            # Create instance and fit
            kmeans = kmeans_class(n_clusters=n_clusters)
            kmeans.fit(X)
            
            # Basic validation checks
            if not hasattr(kmeans, 'centroids') or kmeans.centroids is None:
                return {'passed': False, 'message': "Centroids not properly initialized or updated"}
            
            if kmeans.centroids.shape != (n_clusters, X.shape[1]):
                return {
                    'passed': False, 
                    'message': f"Expected centroids shape {(n_clusters, X.shape[1])}, got {kmeans.centroids.shape}"
                }
            
            # Check if labels are assigned to all points
            if not hasattr(kmeans, 'labels') or kmeans.labels is None:
                return {'passed': False, 'message': "Cluster labels not properly assigned"}
            
            if len(kmeans.labels) != X.shape[0]:
                return {
                    'passed': False, 
                    'message': f"Expected labels for {X.shape[0]} points, got {len(kmeans.labels)}"
                }
            
            # Check if all clusters are used
            unique_clusters = np.unique(kmeans.labels)
            if len(unique_clusters) < n_clusters:
                return {
                    'passed': False, 
                    'message': f"Expected {n_clusters} clusters, but only {len(unique_clusters)} were used"
                }
            
            # Check if centroids are reasonable (not too far from data points)
            max_dist_to_centroid = 0
            for i in range(X.shape[0]):
                point = X[i]
                centroid = kmeans.centroids[kmeans.labels[i]]
                dist = np.linalg.norm(point - centroid)
                max_dist_to_centroid = max(max_dist_to_centroid, dist)
            
            # Heuristic: max distance to centroid shouldn't be more than 3x the max distance between any two points
            max_dist_between_points = 0
            for i in range(min(100, X.shape[0])):  # Sample to avoid O(n²) computation
                for j in range(i+1, min(100, X.shape[0])):
                    dist = np.linalg.norm(X[i] - X[j])
                    max_dist_between_points = max(max_dist_between_points, dist)
            
            if max_dist_to_centroid > 3 * max_dist_between_points:
                return {
                    'passed': False, 
                    'message': "Centroids seem too far from data points. Check your implementation."
                }
            
            return {'passed': True, 'message': "K-means implementation passed validation checks"}
            
        except Exception as e:
            return {'passed': False, 'message': f"Error during validation: {str(e)}"}
    
    def _nn_validation(self, test_case: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, bool]:
        """Validate neural network implementation."""
        try:
            # Find the neural network class
            nn_class = None
            for var_name, var_value in local_vars.items():
                if var_name.endswith('NeuralNetwork') and callable(getattr(var_value, 'fit', None)):
                    nn_class = var_value
                    break
            
            if nn_class is None:
                return {'passed': False, 'message': "Could not find neural network class in your code"}
            
            # Get test data
            input_data = test_case['input']
            X_train = np.array(input_data['X_train'])
            y_train = np.array(input_data['y_train'])
            X_test = np.array(input_data['X_test'])
            hidden_size = input_data.get('hidden_size', 4)
            epochs = input_data.get('epochs', 5000)
            
            # Create instance and train
            nn = nn_class(
                input_size=X_train.shape[1], 
                hidden_size=hidden_size, 
                output_size=y_train.shape[1]
            )
            
            # Train with a timeout to prevent infinite loops
            training_timeout = 60  # seconds
            start_time = time.time()
            
            def train_with_timeout():
                nn.train(X_train, y_train, epochs=epochs)
            
            training_thread = threading.Thread(target=train_with_timeout)
            training_thread.daemon = True
            training_thread.start()
            training_thread.join(timeout=training_timeout)
            
            if training_thread.is_alive():
                return {'passed': False, 'message': f"Training took too long (> {training_timeout}s). Check for infinite loops."}
            
            # Make predictions
            predictions = nn.predict(X_test)
            
            # For binary classification, check if predictions match expected shape
            if predictions.shape != y_train.shape:
                return {
                    'passed': False, 
                    'message': f"Expected predictions shape {y_train.shape}, got {predictions.shape}"
                }
            
            # For XOR problem, check accuracy
            accuracy = np.mean((predictions > 0.5).astype(int) == y_train)
            
            if accuracy < 0.75:
                return {
                    'passed': False, 
                    'message': f"Model accuracy is too low: {accuracy:.2f}. The network should learn the pattern."
                }
            
            return {'passed': True, 'message': f"Neural network implementation passed with accuracy {accuracy:.2f}"}
            
        except Exception as e:
            return {'passed': False, 'message': f"Error during validation: {str(e)}"}
    
    def _anomaly_validation(self, test_case: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, bool]:
        """Validate anomaly detection implementation."""
        try:
            # Find the anomaly detector class
            detector_class = None
            for var_name, var_value in local_vars.items():
                if var_name.endswith('Detector') and callable(getattr(var_value, 'fit', None)):
                    detector_class = var_value
                    break
            
            if detector_class is None:
                return {'passed': False, 'message': "Could not find anomaly detector class in your code"}
            
            # Get test data
            input_data = test_case['input']
            
            # Generate data based on input parameters
            np.random.seed(42)  # For reproducibility
            normal_data = np.random.multivariate_normal(
                mean=input_data['normal_mean'], 
                cov=input_data['normal_cov'], 
                size=input_data['normal_size']
            )
            anomalies = np.random.multivariate_normal(
                mean=input_data['anomaly_mean'], 
                cov=input_data['anomaly_cov'], 
                size=input_data['anomaly_size']
            )
            
            # Combine data
            X = np.vstack([normal_data, anomalies])
            y_true = np.zeros(X.shape[0])
            y_true[input_data['normal_size']:] = 1  # Mark anomalies
            
            # Create detector and train on normal data only
            detector = detector_class()
            detector.fit(normal_data)
            
            # Evaluate
            if hasattr(detector, 'evaluate') and callable(detector.evaluate):
                eval_result = detector.evaluate(X, y_true)
                
                # Check if evaluation returns reasonable results
                if isinstance(eval_result, dict) and 'precision_recall_auc' in eval_result:
                    pr_auc = eval_result['precision_recall_auc']
                    
                    # For well-separated clusters, AUC should be high
                    if 'well-separated' in test_case['description'] and pr_auc < 0.8:
                        return {
                            'passed': False, 
                            'message': f"Precision-Recall AUC is too low: {pr_auc:.2f} for well-separated clusters"
                        }
                    
                    # For overlapping clusters, AUC should be reasonable
                    if 'overlapping' in test_case['description'] and pr_auc < 0.6:
                        return {
                            'passed': False, 
                            'message': f"Precision-Recall AUC is too low: {pr_auc:.2f} for overlapping clusters"
                        }
                    
                    return {
                        'passed': True, 
                        'message': f"Anomaly detection passed with Precision-Recall AUC: {pr_auc:.2f}"
                    }
            
            # If no evaluate method or it doesn't return expected metrics, check predictions directly
            predictions, _ = detector.predict(X)
            
            # Calculate basic metrics
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Check if metrics are reasonable
            if 'well-separated' in test_case['description'] and f1 < 0.7:
                return {
                    'passed': False, 
                    'message': f"F1 score is too low: {f1:.2f} for well-separated clusters"
                }
            
            if 'overlapping' in test_case['description'] and f1 < 0.5:
                return {
                    'passed': False, 
                    'message': f"F1 score is too low: {f1:.2f} for overlapping clusters"
                }
            
            return {'passed': True, 'message': f"Anomaly detection passed with F1 score: {f1:.2f}"}
            
        except Exception as e:
            return {'passed': False, 'message': f"Error during validation: {str(e)}"}
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get a scenario by ID."""
        return self.scenarios.get(scenario_id)
    
    def list_scenarios(self, difficulty: str = None, category: str = None) -> List[Dict[str, Any]]:
        """List available scenarios, optionally filtered by difficulty or category."""
        result = []
        
        for scenario_id, scenario in self.scenarios.items():
            if difficulty and scenario.difficulty != difficulty:
                continue
            if category and scenario.category != category:
                continue
                
            result.append({
                'id': scenario_id,
                'title': scenario.title,
                'description': scenario.description,
                'difficulty': scenario.difficulty,
                'category': scenario.category
            })
            
        return result
    
    def get_learning_path(self, path_id: str) -> List[Dict[str, Any]]:
        """Get a predefined learning path with ordered scenarios."""
        paths = self._create_learning_paths()
        
        if path_id not in paths:
            return []
            
        result = []
        for scenario_id in paths[path_id]:
            scenario = self.scenarios.get(scenario_id)
            if scenario:
                result.append({
                    'id': scenario_id,
                    'title': scenario.title,
                    'description': scenario.description,
                    'difficulty': scenario.difficulty,
                    'category': scenario.category
                })
                
        return result
    
    def track_attempt(self, scenario_id: str, user_id: str, code: str, result: SandboxResult) -> None:
        """Track a user's attempt at solving a scenario."""
        if user_id not in self.user_attempts:
            self.user_attempts[user_id] = []
            
        attempt = ScenarioAttempt(
            scenario_id=scenario_id,
            user_id=user_id,
            code_submitted=code,
            result=result,
            completed=result.success,
            iterations=len(self.user_attempts.get(user_id, [])) + 1
        )
        
        self.user_attempts[user_id].append(attempt)
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get a user's progress across all scenarios."""
        if user_id not in self.user_attempts:
            return {
                'scenarios_attempted': 0,
                'scenarios_completed': 0,
                'total_attempts': 0,
                'scenario_status': {}
            }
            
        attempts = self.user_attempts[user_id]
        scenario_status = {}
        
        for attempt in attempts:
            if attempt.scenario_id not in scenario_status:
                scenario_status[attempt.scenario_id] = {
                    'attempts': 0,
                    'completed': False,
                    'last_attempt_timestamp': None
                }
                
            status = scenario_status[attempt.scenario_id]
            status['attempts'] += 1
            status['completed'] = status['completed'] or attempt.completed
            status['last_attempt_timestamp'] = attempt.timestamp
            
        return {
            'scenarios_attempted': len(scenario_status),
            'scenarios_completed': sum(1 for s in scenario_status.values() if s['completed']),
            'total_attempts': len(attempts),
            'scenario_status': scenario_status
        }
    
    def generate_challenge(self, difficulty: str = 'intermediate') -> Scenario:
        """Generate a new challenge based on existing scenarios but with modified parameters."""
        # Select a random scenario of the specified difficulty
        candidates = [s for s in self.scenarios.values() if s.difficulty == difficulty]
        
        if not candidates:
            # Fall back to any difficulty if no scenarios match the requested difficulty
            candidates = list(self.scenarios.values())
            
        base_scenario = random.choice(candidates)
        
        # Create a new scenario ID
        new_id = f"{base_scenario.id}_challenge_{uuid.uuid4().hex[:8]}"
        
        # Modify the scenario to create a challenge
        if base_scenario.category == 'tree_based_models':
            return self._generate_tree_challenge(base_scenario, new_id)
        elif base_scenario.category == 'clustering':
            return self._generate_clustering_challenge(base_scenario, new_id)
        elif base_scenario.category == 'neural_networks':
            return self._generate_nn_challenge(base_scenario, new_id)
        elif base_scenario.category == 'applied_ml':
            return self._generate_applied_ml_challenge(base_scenario, new_id)
        else:
                        # Default: just return a copy of the base scenario with a new ID
            modified = Scenario(
                id=new_id,
                title=f"Challenge: {base_scenario.title}",
                description=f"A challenging variant of: {base_scenario.description}",
                difficulty=difficulty,
                category=base_scenario.category,
                instructions=base_scenario.instructions,
                starter_code=base_scenario.starter_code,
                test_cases=base_scenario.test_cases,
                evaluation_criteria=base_scenario.evaluation_criteria,
                hints=base_scenario.hints,
                solution_code=base_scenario.solution_code
            )
            
            return modified
    
    def _generate_tree_challenge(self, base_scenario: Scenario, new_id: str) -> Scenario:
        """Generate a modified tree-based model challenge."""
        # Create more complex test cases
        new_test_cases = []
        
        for test_case in base_scenario.test_cases:
            # Make a copy of the test case
            new_test = dict(test_case)
            
            # Modify input data if it exists
            if 'input' in new_test:
                input_data = dict(new_test['input'])
                
                # Add more features or samples if X_train exists
                if 'X_train' in input_data:
                    X_train = np.array(input_data['X_train'])
                    y_train = np.array(input_data['y_train'])
                    
                    # Add noise features
                    n_samples = X_train.shape[0]
                    n_features = X_train.shape[1]
                    noise_features = np.random.randn(n_samples, 2)  # Add 2 noise features
                    
                    X_train_new = np.hstack([X_train, noise_features])
                    
                    # Update test case
                    input_data['X_train'] = X_train_new.tolist()
                    
                    # Also update X_test if it exists
                    if 'X_test' in input_data:
                        X_test = np.array(input_data['X_test'])
                        n_test = X_test.shape[0]
                        noise_test = np.random.randn(n_test, 2)
                        X_test_new = np.hstack([X_test, noise_test])
                        input_data['X_test'] = X_test_new.tolist()
                
                # Increase max_depth if it exists
                if 'max_depth' in input_data:
                    input_data['max_depth'] += 2
                    
                new_test['input'] = input_data
            
            new_test_cases.append(new_test)
            
        # Add a new test case with non-linear decision boundary
        if base_scenario.id == 'decision_tree_basic':
            # Create a spiral dataset
            n_samples = 100
            noise = 0.1
            
            def make_spiral(n_samples, noise):
                n = np.sqrt(np.random.rand(n_samples)) * 780 * (2*np.pi)/360
                d1x = -np.cos(n) * n + np.random.rand(n_samples) * noise
                d1y = np.sin(n) * n + np.random.rand(n_samples) * noise
                return np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
            
            spiral_data = make_spiral(n_samples, noise)
            X_spiral = spiral_data[:, :2]
            y_spiral = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
            
            # Split into train and test
            indices = np.random.permutation(len(X_spiral))
            train_idx, test_idx = indices[:int(0.8*len(indices))], indices[int(0.8*len(indices)):]
            
            X_train = X_spiral[train_idx]
            y_train = y_spiral[train_idx]
            X_test = X_spiral[test_idx]
            y_test = y_spiral[test_idx]
            
            new_test_cases.append({
                'input': {
                    'X_train': X_train.tolist(),
                    'y_train': y_train.tolist(),
                    'X_test': X_test.tolist(),
                    'max_depth': 10
                },
                'validation': 'tree_validation',
                'description': 'Spiral dataset with non-linear decision boundary'
            })
        
        # Create the modified scenario
        modified = Scenario(
            id=new_id,
            title=f"Challenge: {base_scenario.title}",
            description=f"A more challenging variant of: {base_scenario.description}",
            difficulty='advanced',  # Increase difficulty
            category=base_scenario.category,
            instructions=base_scenario.instructions + [
                "This challenge includes datasets with noise features and non-linear decision boundaries."
            ],
            starter_code=base_scenario.starter_code,
            test_cases=new_test_cases,
            evaluation_criteria=base_scenario.evaluation_criteria + [
                "Handling of noise features",
                "Performance on complex non-linear boundaries"
            ],
            hints=base_scenario.hints + [
                "Consider using feature importance to identify relevant features",
                "For non-linear boundaries, you may need to increase tree depth"
            ],
            solution_code=base_scenario.solution_code
        )
        
        return modified
    
    def _generate_clustering_challenge(self, base_scenario: Scenario, new_id: str) -> Scenario:
        """Generate a modified clustering challenge."""
        # Create more complex test cases
        new_test_cases = []
        
        for test_case in base_scenario.test_cases:
            # Make a copy of the test case
            new_test = dict(test_case)
            
            # Modify input data if it exists
            if 'input' in new_test:
                input_data = dict(new_test['input'])
                
                # Increase number of clusters
                if 'n_clusters' in input_data:
                    input_data['n_clusters'] += 2
                
                # Add more complex data if X exists
                if 'X' in input_data:
                    X = np.array(input_data['X'])
                    n_samples = X.shape[0]
                    
                    # Add more clusters
                    n_new_clusters = 2
                    new_clusters = []
                    
                    for i in range(n_new_clusters):
                        # Create a new cluster center
                        center = np.random.uniform(-10, 10, size=X.shape[1])
                        # Generate points around the center
                        cluster_points = center + np.random.randn(n_samples // 3, X.shape[1])
                        new_clusters.append(cluster_points)
                    
                    # Combine with original data
                    X_new = np.vstack([X] + new_clusters)
                    input_data['X'] = X_new.tolist()
                
                new_test['input'] = input_data
            
            new_test_cases.append(new_test)
        
        # Add a new test case with non-spherical clusters
        if base_scenario.id == 'kmeans_clustering':
            # Create moons dataset
            from sklearn.datasets import make_moons
            X_moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
            
            # Scale and shift to create multiple moons
            X_moons1 = X_moons * 2
            X_moons2 = X_moons * 2 + np.array([5, 5])
            X_moons3 = X_moons * 2 + np.array([-5, 5])
            
            X_combined = np.vstack([X_moons1, X_moons2, X_moons3])
            
            new_test_cases.append({
                'input': {
                    'X': X_combined.tolist(),
                    'n_clusters': 6  # Challenging: need to find 6 half-moons
                },
                'validation': 'cluster_validation',
                'description': 'Non-spherical clusters (moons)'
            })
        
        # Create the modified scenario
        modified = Scenario(
            id=new_id,
            title=f"Challenge: {base_scenario.title}",
            description=f"A more challenging variant of: {base_scenario.description}",
            difficulty='advanced',  # Increase difficulty
            category=base_scenario.category,
            instructions=base_scenario.instructions + [
                "This challenge includes datasets with non-spherical clusters and varying densities."
            ],
            starter_code=base_scenario.starter_code,
            test_cases=new_test_cases,
            evaluation_criteria=base_scenario.evaluation_criteria + [
                "Handling of non-spherical clusters",
                "Robustness to varying cluster densities"
            ],
            hints=base_scenario.hints + [
                "K-means works best with spherical clusters of similar size",
                "Consider using a different initialization strategy",
                "You might need to run K-means multiple times with different initializations"
            ],
            solution_code=base_scenario.solution_code
        )
        
        return modified
    
    def _generate_nn_challenge(self, base_scenario: Scenario, new_id: str) -> Scenario:
        """Generate a modified neural network challenge."""
        # Create more complex test cases
        new_test_cases = []
        
        for test_case in base_scenario.test_cases:
            # Make a copy of the test case
            new_test = dict(test_case)
            
            # Modify input data if it exists
            if 'input' in new_test:
                input_data = dict(new_test['input'])
                
                # Increase hidden size and epochs
                if 'hidden_size' in input_data:
                    input_data['hidden_size'] += 4
                if 'epochs' in input_data:
                    input_data['epochs'] = int(input_data['epochs'] * 1.5)
                
                # Add more complex data if X_train exists
                if 'X_train' in input_data:
                    X_train = np.array(input_data['X_train'])
                    y_train = np.array(input_data['y_train'])
                    
                    # Add noise features
                    n_samples = X_train.shape[0]
                    noise_features = np.random.randn(n_samples, 2)  # Add 2 noise features
                    
                    X_train_new = np.hstack([X_train, noise_features])
                    
                    # Update test case
                    input_data['X_train'] = X_train_new.tolist()
                    
                    # Also update X_test if it exists
                    if 'X_test' in input_data:
                        X_test = np.array(input_data['X_test'])
                        n_test = X_test.shape[0]
                        noise_test = np.random.randn(n_test, 2)
                        X_test_new = np.hstack([X_test, noise_test])
                        input_data['X_test'] = X_test_new.tolist()
                
                new_test['input'] = input_data
            
            new_test_cases.append(new_test)
        
        # Add a new test case with a more complex function
        if base_scenario.id == 'neural_network_basic':
            # Create a dataset for learning a sine function
            X_sine = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
            y_sine = np.sin(X_sine)
            
            # Split into train and test
            indices = np.random.permutation(len(X_sine))
            train_idx, test_idx = indices[:int(0.8*len(indices))], indices[int(0.8*len(indices)):]
            
            X_train = X_sine[train_idx]
            y_train = y_sine[train_idx]
            X_test = X_sine[test_idx]
            y_test = y_sine[test_idx]
            
            new_test_cases.append({
                'input': {
                    'X_train': X_train.tolist(),
                    'y_train': y_train.tolist(),
                    'X_test': X_test.tolist(),
                    'hidden_size': 10,
                    'epochs': 10000
                },
                'validation': 'nn_regression_validation',
                'description': 'Learning a sine function (regression)'
            })
        
        # Create the modified scenario
        modified = Scenario(
            id=new_id,
            title=f"Challenge: {base_scenario.title}",
            description=f"A more challenging variant of: {base_scenario.description}",
            difficulty='advanced',  # Increase difficulty
            category=base_scenario.category,
            instructions=base_scenario.instructions + [
                "This challenge includes regression tasks and datasets with noise features."
            ],
            starter_code=base_scenario.starter_code,
            test_cases=new_test_cases,
            evaluation_criteria=base_scenario.evaluation_criteria + [
                "Handling of regression tasks",
                "Robustness to noise features"
            ],
            hints=base_scenario.hints + [
                "For regression tasks, consider using a linear activation in the output layer",
                "You may need to normalize inputs for better convergence",
                "Consider adding regularization to handle noise features"
            ],
            solution_code=base_scenario.solution_code
        )
        
        return modified
    
    def _generate_applied_ml_challenge(self, base_scenario: Scenario, new_id: str) -> Scenario:
        """Generate a modified applied ML challenge."""
        # Create more complex test cases based on the specific scenario
        if base_scenario.id == 'recommendation_system':
            return self._generate_recommendation_challenge(base_scenario, new_id)
        elif base_scenario.id == 'anomaly_detection':
            return self._generate_anomaly_challenge(base_scenario, new_id)
        else:
            # Default modification
            return Scenario(
                id=new_id,
                title=f"Challenge: {base_scenario.title}",
                description=f"A more challenging variant of: {base_scenario.description}",
                difficulty='advanced',
                category=base_scenario.category,
                instructions=base_scenario.instructions,
                starter_code=base_scenario.starter_code,
                test_cases=base_scenario.test_cases,
                evaluation_criteria=base_scenario.evaluation_criteria,
                hints=base_scenario.hints,
                solution_code=base_scenario.solution_code
            )
    
    def _generate_recommendation_challenge(self, base_scenario: Scenario, new_id: str) -> Scenario:
        """Generate a modified recommendation system challenge."""
        # Create more complex test cases
        new_test_cases = []
        
        for test_case in base_scenario.test_cases:
            # Make a copy of the test case
            new_test = dict(test_case)
            
            # Modify input data if it exists
            if 'input' in new_test:
                input_data = dict(new_test['input'])
                
                # Add more items and features if items_df exists
                if 'items_df' in input_data:
                    items_df = input_data['items_df']
                                        # Add more items
                    n_existing = len(items_df['item_id'])
                    n_new = 5
                    
                    # Create new items
                    new_items = {
                        'item_id': list(range(n_existing + 1, n_existing + n_new + 1)),
                        'title': [f"New Item {i}" for i in range(1, n_new + 1)],
                        'category': ['electronics', 'books', 'clothing', 'home', 'sports'],
                        'description': [
                            "A cutting-edge electronic gadget with AI features",
                            "A bestselling novel about machine learning",
                            "High-performance athletic wear",
                            "Smart home automation device",
                            "Professional sports equipment"
                        ]
                    }
                    
                    # Combine with existing items
                    for key in items_df:
                        items_df[key] = items_df[key] + new_items.get(key, [])
                    
                    # Add a new feature column
                    if 'tags' not in items_df:
                        items_df['tags'] = [
                            "popular tech gadget",
                            "budget friendly tech",
                            "educational reading",
                            "fiction bestseller",
                            "casual fashion",
                            "tech gadget premium",
                            "educational tech book",
                            "premium sportswear",
                            "smart home essential",
                            "professional equipment"
                        ]
                        
                        # Add tags to feature columns
                        if 'feature_columns' in input_data:
                            input_data['feature_columns'] = input_data['feature_columns'] + ['tags']
                    
                    input_data['items_df'] = items_df
                
                new_test['input'] = input_data
            
            new_test_cases.append(new_test)
        
        # Add a new test case with cold start problem
        new_test_cases.append({
            'input': {
                'items_df': {
                    'item_id': list(range(1, 11)),
                    'title': [f"Item {i}" for i in range(1, 11)],
                    'category': ['electronics', 'electronics', 'books', 'books', 'clothing',
                                'electronics', 'books', 'clothing', 'home', 'sports'],
                    'description': [
                        "A great electronic device", "Another electronic item", 
                        "An interesting book", "A classic novel", "Stylish clothing",
                        "A cutting-edge electronic gadget", "A bestselling novel",
                        "High-performance athletic wear", "Smart home automation device",
                        "Professional sports equipment"
                    ],
                    'tags': [
                        "popular tech gadget", "budget friendly tech",
                        "educational reading", "fiction bestseller", "casual fashion",
                        "tech gadget premium", "educational tech book",
                        "premium sportswear", "smart home essential", "professional equipment"
                    ]
                },
                'feature_columns': ['category', 'description', 'tags'],
                'query_item': 10,  # New item with few interactions
                'n': 3
            },
            'expected_output': [8, 7, 5],  # Similar items based on content
            'description': 'Cold start recommendation for a new item'
        })
        
        # Create the modified scenario
        modified = Scenario(
            id=new_id,
            title=f"Challenge: {base_scenario.title}",
            description=f"A more challenging variant with cold start problems and multiple features",
            difficulty='advanced',
            category=base_scenario.category,
            instructions=base_scenario.instructions + [
                "This challenge includes handling cold start problems and weighting multiple features."
            ],
            starter_code=base_scenario.starter_code,
            test_cases=new_test_cases,
            evaluation_criteria=base_scenario.evaluation_criteria + [
                "Handling of cold start problems",
                "Effective weighting of multiple features"
            ],
            hints=base_scenario.hints + [
                "For cold start problems, rely more heavily on content features",
                "Consider giving different weights to different feature types",
                "Tags can be particularly useful for finding similar items"
            ],
            solution_code=base_scenario.solution_code
        )
        
        return modified
    
    def _generate_anomaly_challenge(self, base_scenario: Scenario, new_id: str) -> Scenario:
        """Generate a modified anomaly detection challenge."""
        # Create more complex test cases
        new_test_cases = []
        
        for test_case in base_scenario.test_cases:
            # Make a copy of the test case
            new_test = dict(test_case)
            
            # Modify input data if it exists
            if 'input' in new_test:
                input_data = dict(new_test['input'])
                
                # Make anomalies harder to detect
                if 'anomaly_mean' in input_data and 'normal_mean' in input_data:
                    # Move anomaly mean closer to normal mean
                    normal_mean = np.array(input_data['normal_mean'])
                    anomaly_mean = np.array(input_data['anomaly_mean'])
                    
                    # Move 30% closer
                    new_anomaly_mean = anomaly_mean - 0.3 * (anomaly_mean - normal_mean)
                    input_data['anomaly_mean'] = new_anomaly_mean.tolist()
                
                # Increase covariance of normal data
                if 'normal_cov' in input_data:
                    normal_cov = np.array(input_data['normal_cov'])
                    # Increase variance
                    input_data['normal_cov'] = (normal_cov * 1.5).tolist()
                
                new_test['input'] = input_data
            
            new_test_cases.append(new_test)
        
        # Add a new test case with multimodal normal distribution
        new_test_cases.append({
            'input': {
                'normal_mean1': [0, 0],
                'normal_cov1': [[1, 0], [0, 1]],
                'normal_size1': 500,
                'normal_mean2': [4, 0],
                'normal_cov2': [[1, 0], [0, 1]],
                'normal_size2': 500,
                'anomaly_mean': [2, 4],
                'anomaly_cov': [[0.5, 0], [0, 0.5]],
                'anomaly_size': 50
            },
            'validation': 'multimodal_anomaly_validation',
            'description': 'Multimodal normal distribution with anomalies'
        })
        
        # Create the modified scenario
        modified = Scenario(
            id=new_id,
            title=f"Challenge: {base_scenario.title}",
            description=f"A more challenging variant with multimodal distributions and subtle anomalies",
            difficulty='advanced',
            category=base_scenario.category,
            instructions=base_scenario.instructions + [
                "This challenge includes multimodal normal distributions and subtle anomalies."
            ],
            starter_code=base_scenario.starter_code,
            test_cases=new_test_cases,
            evaluation_criteria=base_scenario.evaluation_criteria + [
                "Handling of multimodal distributions",
                "Detection of subtle anomalies"
            ],
            hints=base_scenario.hints + [
                "Consider using mixture models for multimodal distributions",
                "Feature engineering can help detect subtle anomalies",
                "Try different distance metrics for anomaly scoring"
            ],
            solution_code=base_scenario.solution_code
        )
        
        return modified
    
    def _multimodal_anomaly_validation(self, test_case: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, bool]:
        """Validate anomaly detection with multimodal normal distribution."""
        try:
            # Find the anomaly detector class
            detector_class = None
            for var_name, var_value in local_vars.items():
                if var_name.endswith('Detector') and callable(getattr(var_value, 'fit', None)):
                    detector_class = var_value
                    break
            
            if detector_class is None:
                return {'passed': False, 'message': "Could not find anomaly detector class in your code"}
            
            # Get test data
            input_data = test_case['input']
            
            # Generate data based on input parameters
            np.random.seed(42)  # For reproducibility
            
            # Generate multimodal normal data
            normal_data1 = np.random.multivariate_normal(
                mean=input_data['normal_mean1'], 
                cov=input_data['normal_cov1'], 
                size=input_data['normal_size1']
            )
            
            normal_data2 = np.random.multivariate_normal(
                mean=input_data['normal_mean2'], 
                cov=input_data['normal_cov2'], 
                size=input_data['normal_size2']
            )
            
            normal_data = np.vstack([normal_data1, normal_data2])
            
            # Generate anomalies
            anomalies = np.random.multivariate_normal(
                mean=input_data['anomaly_mean'], 
                cov=input_data['anomaly_cov'], 
                size=input_data['anomaly_size']
            )
            
            # Combine data
            X = np.vstack([normal_data, anomalies])
            y_true = np.zeros(X.shape[0])
            y_true[input_data['normal_size1'] + input_data['normal_size2']:] = 1  # Mark anomalies
            
            # Create detector and train on normal data only
            detector = detector_class()
            detector.fit(normal_data)
            
            # Evaluate
            if hasattr(detector, 'evaluate') and callable(detector.evaluate):
                eval_result = detector.evaluate(X, y_true)
                
                # Check if evaluation returns reasonable results
                if isinstance(eval_result, dict) and 'precision_recall_auc' in eval_result:
                    pr_auc = eval_result['precision_recall_auc']
                    
                    # For multimodal data, AUC should be reasonable
                    if pr_auc < 0.6:
                        return {
                            'passed': False, 
                            'message': f"Precision-Recall AUC is too low: {pr_auc:.2f} for multimodal data"
                        }
                    
                    return {
                        'passed': True, 
                        'message': f"Anomaly detection passed with Precision-Recall AUC: {pr_auc:.2f}"
                    }
            
            # If no evaluate method or it doesn't return expected metrics, check predictions directly
            predictions, _ = detector.predict(X)
            
            # Calculate basic metrics
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Check if metrics are reasonable
            if f1 < 0.5:
                return {
                    'passed': False, 
                    'message': f"F1 score is too low: {f1:.2f} for multimodal data"
                }
            
            return {'passed': True, 'message': f"Anomaly detection passed with F1 score: {f1:.2f}"}
            
        except Exception as e:
            return {'passed': False, 'message': f"Error during validation: {str(e)}"}
    
    def _nn_regression_validation(self, test_case: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, bool]:
        """Validate neural network regression implementation."""
        try:
            # Find the neural network class
            nn_class = None
            for var_name, var_value in local_vars.items():
                if var_name.endswith('NeuralNetwork') and callable(getattr(var_value, 'fit', None)):
                    nn_class = var_value
                    break
            
            if nn_class is None:
                return {'passed': False, 'message': "Could not find neural network class in your code"}
            
            # Get test data
            input_data = test_case['input']
            X_train = np.array(input_data['X_train'])
            y_train = np.array(input_data['y_train'])
            X_test = np.array(input_data['X_test'])
            hidden_size = input_data.get('hidden_size', 10)
            epochs = input_data.get('epochs', 10000)
            
            # Create instance and train
            nn = nn_class(
                input_size=X_train.shape[1], 
                hidden_size=hidden_size, 
                output_size=y_train.shape[1] if len(y_train.shape) > 1 else 1
            )
            
            # Train with a timeout to prevent infinite loops
            training_timeout = 60  # seconds
            start_time = time.time()
            
            def train_with_timeout():
                nn.train(X_train, y_train, epochs=epochs)
            
            training_thread = threading.Thread(target=train_with_timeout)
            training_thread.daemon = True
            training_thread.start()
            training_thread.join(timeout=training_timeout)
            
            if training_thread.is_alive():
                return {'passed': False, 'message': f"Training took too long (> {training_timeout}s). Check for infinite loops."}
            
            # Make predictions
            predictions = nn.predict(X_test)
            
            # For regression, check mean squared error
            y_test = np.array(input_data.get('y_test', y_train))  # Use y_train as y_test if not provided
            mse = np.mean((predictions - y_test) ** 2)
            
            # For sine function, MSE should be reasonably low
            if 'sine' in test_case['description'].lower() and mse > 0.1:
                return {
                    'passed': False, 
                    'message': f"Mean squared error is too high: {mse:.4f}. The network should fit the sine function better."
                }
            
            return {'passed': True, 'message': f"Neural network regression passed with MSE: {mse:.4f}"}
            
        except Exception as e:
            return {'passed': False, 'message': f"Error during validation: {str(e)}"}
    def _tree_validation(self, test_case: Dict[str, Any], local_vars: Dict[str, Any]) -> Dict[str, bool]:
        """Validate decision tree implementation on complex datasets."""
        try:
            # Find the decision tree classifier class
            tree_class = None
            for var_name, var_value in local_vars.items():
                if var_name.endswith('Classifier') and callable(getattr(var_value, 'fit', None)):
                    tree_class = var_value
                    break
            
            if tree_class is None:
                return {'passed': False, 'message': "Could not find decision tree classifier class in your code"}
            
            # Get test data
            input_data = test_case['input']
            X_train = np.array(input_data['X_train'])
            y_train = np.array(input_data['y_train'])
            X_test = np.array(input_data['X_test'])
            max_depth = input_data.get('max_depth', 5)
            
            # Create instance and train
            tree = tree_class(max_depth=max_depth)
            tree.fit(X_train, y_train)
            
            # Make predictions
            predictions = tree.predict(X_test)
            
            # For spiral dataset, check accuracy
            if 'spiral' in test_case['description'].lower():
                y_test = np.array(input_data.get('y_test', y_train))  # Use y_train as y_test if not provided
                accuracy = np.mean(predictions == y_test)
                
                if accuracy < 0.7:
                    return {
                        'passed': False, 
                        'message': f"Accuracy is too low: {accuracy:.2f}. The tree should fit the spiral pattern better."
                    }
                
                return {'passed': True, 'message': f"Decision tree passed with accuracy: {accuracy:.2f}"}
            
            # For other datasets, check if tree structure is reasonable
            if not hasattr(tree, 'tree') or tree.tree is None:
                return {'passed': False, 'message': "Tree structure not properly built"}
            
            # Check if tree has reasonable depth
            max_actual_depth = 0
            
            def get_max_depth(node, current_depth=0):
                nonlocal max_actual_depth
                if node is None:
                    return
                
                max_actual_depth = max(max_actual_depth, current_depth)
                
                # If node has children, check them
                if hasattr(node, 'left') and node.left is not None:
                    get_max_depth(node.left, current_depth + 1)
                if hasattr(node, 'right') and node.right is not None:
                    get_max_depth(node.right, current_depth + 1)
            
            # Start from root node
            if hasattr(tree, 'tree'):
                get_max_depth(tree.tree)
            
            if max_actual_depth < 3:
                return {
                    'passed': False, 
                    'message': f"Tree depth is too shallow: {max_actual_depth}. Complex patterns require deeper trees."
                }
            
            return {'passed': True, 'message': f"Decision tree passed with depth: {max_actual_depth}"}
            
        except Exception as e:
            return {'passed': False, 'message': f"Error during validation: {str(e)}"}
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class Scenario:
    """Represents a machine learning scenario for the sandbox environment."""
    
    id: str
    title: str
    description: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    category: str  # 'tree_based_models', 'clustering', 'neural_networks', 'applied_ml'
    instructions: List[str]
    starter_code: str
    test_cases: List[Dict[str, Any]]
    evaluation_criteria: List[str]
    hints: List[str]
    solution_code: str = ""

@dataclass
class SandboxResult:
    """Represents the result of executing code in the sandbox."""
    
    success: bool
    output: str
    error: str
    execution_time_ms: float
    memory_used_mb: float
    feedback: str

@dataclass
class ScenarioAttempt:
    """Represents a user's attempt at solving a scenario."""
    
    scenario_id: str
    user_id: str
    code_submitted: str
    result: SandboxResult
    completed: bool
    iterations: int
    timestamp: datetime = field(default_factory=datetime.now)

                    
