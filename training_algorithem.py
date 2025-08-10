from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
# Importing the necessary libraries for decision tree training
def train_decision_tree(X, y, max_depth=10):
    """
    This function trains a Decision Tree classifier on the provided HOG feature vectors and labels. 
    It splits the dataset into training and testing sets, trains the model, evaluates its accuracy, 
    and prints the decision rules for interpretability. which makes it interpretable and easy to understand.
    """
    X = np.array(X) #  this converts the list of the feature vectors into numpy arrays, as our machine learning algorithem uses numpy arrays for computations.
    y = np.array(y) #  this converts the list of labels into numpy arrrays.

    X_train, X_test, y_train, y_test = train_test_split(   # here we split our data set into training and testing sets
        X, y, test_size=0.2, random_state=42, stratify=y
    )               #  we use stratify=y to ensure that the class distribution is preserved in both training and testing sets.

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42) # This initializes the Decision Tree classifier with a maximum depth for the tree to prevent overfitting.
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) # This uses the trained model to make predictions on the test set.
    acc = accuracy_score(y_test, y_pred) # This calculates the accuracy of the model by comparing the predicted labels with the true labels from the test set.
    print(f" Decision Tree Accuracy: {acc*100:.2f}%")

    rules = export_text(model, max_depth=2, feature_names=[f'f{i}' for i in range(X.shape[1])])
    print("\n Sample Tree Rules (showing depth 2 only):\n")
    print(rules) # This prints the decision rules of the model, this will help to understand why model predicted a certain name for a given input, which makes it diffirent from conventional black-box models like neural networks.

    return model
