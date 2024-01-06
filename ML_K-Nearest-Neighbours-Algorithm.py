import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def visualize_knn(x_train, y_train, x_test, y_pred):
    plt.figure(figsize=(10, 6))

    # Plot training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis', label='Training Data')

    # Plot test points with predictions
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap='viridis', marker='x', s=150, label='Predictions')

    plt.title('K-Nearest Neighbors Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate random training and test data
    np.random.seed(42)
    x_data = np.random.rand(100, 2)
    y_data = np.concatenate([np.zeros(50), np.ones(50)])  # Balanced labels

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Create and train K-Nearest Neighbors classifier
    k_value = 3
    knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
    knn_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = knn_classifier.predict(x_test)

    # Calculate and print accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")

    # Visualize the KNN predictions
    visualize_knn(x_train, y_train, x_test, y_pred)
