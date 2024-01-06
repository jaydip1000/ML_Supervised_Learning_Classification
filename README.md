# ML_Supervised_Learning_Classification
Implementation of the Naive Bayes and K-Nearest Neighbours algorithms.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_naive_bayes(x_train, y_train):
    '''
    Fit a Naive Bayes classifier using the given training data.
    Parameters:
    - x_train: Features (numpy array or pandas DataFrame)
    - y_train: Output labels (numpy array or pandas Series)
    Returns:
    - prior_prob: Dictionary of class priors
    - conditional_prob: Dictionary of conditional probabilities
    '''


    # Calculate class priors

    unique_classes, class_counts = np.unique(y_train, return_counts=True)

    prior_prob = dict(zip(unique_classes, class_counts / len(y_train)))



    # Calculate conditional probabilities

    conditional_prob = {}

    for label in prior_prob:

        # Extract features for the current class

        subset_x = x_train[y_train == label]


        # Calculate mean and standard deviation for each feature

        mean = np.mean(subset_x, axis=0)
        std = np.std(subset_x, axis=0)

        # Store parameters in conditional_prob dictionary
        
        conditional_prob[label] = {'mean': mean, 'std': std}

    return prior_prob, conditional_prob

def calculate_likelihood(x, mean, std):

    '''
    Calculate the likelihood of a feature value given mean and standard deviation.
    Parameters:
    - x: Feature value
    - mean: Mean of the feature for a specific class
    - std: Standard deviation of the feature for a specific class
    Returns:
    - likelihood: Probability density function value
    '''


    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def naive_bayes_predict(x, prior_prob, conditional_prob):

    '''
    Predict the class labels for given features using Naive Bayes classifier.
    Parameters:
    - x: Features (numpy array or pandas DataFrame)
    - prior_prob: Dictionary of class priors
    - conditional_prob: Dictionary of conditional probabilities
    Returns:
    - predicted_labels: Predicted class labels
    '''



    predicted_labels = []

    for instance in x:

        # Initialize probability for each class

        class_probs = {label: np.log(prior_prob[label]) for label in prior_prob}
        
        
        # Calculate log likelihood for each feature and class

        for label in conditional_prob:

            for i, (mean, std) in enumerate(zip(conditional_prob[label]['mean'], conditional_prob[label]['std'])):

                class_probs[label] += np.log(calculate_likelihood(instance[i], mean, std))



        # Predict the class with the maximum log probability

        predicted_label = max(class_probs, key=class_probs.get)

        predicted_labels.append(predicted_label)



    return np.array(predicted_labels)



# Example usage:

# Assuming you have training data x_train and corresponding labels y_train

# Generate some random training data for demonstration purposes

np.random.seed(42)

data_size = 100

x_train = np.random.randn(data_size, 2)

y_train = np.random.choice([0, 1], size=data_size)



# Fit Naive Bayes classifier

prior_prob, conditional_prob = fit_naive_bayes(x_train, y_train)



# Predict on the training data

y_pred = naive_bayes_predict(x_train, prior_prob, conditional_prob)



# Calculate accuracy

accuracy = np.mean(y_pred == y_train)

print("Accuracy:", accuracy)



# Visualize the training data and decision boundary (for 2D data)

if x_train.shape[1] == 2:

    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', marker='o', label='Actual')

    

    # Decision boundary

    h = .02  # step size in the mesh

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1

    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = naive_bayes_predict(np.c_[xx.ravel(), yy.ravel()], prior_prob, conditional_prob)

    Z = Z.reshape(xx.shape)



    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)

    

    plt.title('Naive Bayes Classifier Decision Boundary')

    plt.xlabel('Feature 1')

    plt.ylabel('Feature 2')

    plt.legend()

    plt.show()
