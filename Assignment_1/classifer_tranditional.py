import torch
import torchvision
import numpy as np
from torchvision import transforms
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def svm_classifier(X_train, y_train, X_test, y_test):

    # Initialize the SVM classifier
    svm_clf = svm.SVC()

    # Train the SVM classifier
    svm_clf.fit(X_train, y_train)

    # Predict labels for the testing set
    svm_y_pred = svm_clf.predict(X_test)

    # Calculate accuracy
    svm_accuracy = accuracy_score(y_test, svm_y_pred)
    print("SVM Accuracy: {:.2f}%".format(svm_accuracy * 100))


def kernel_svm_classifier(X_train, y_train, X_test, y_test):

    # Initialize the Kernel SVM classifier
    kernel_svm_clf = svm.SVC(kernel='rbf')

    # Train the Kernel SVM classifier
    kernel_svm_clf.fit(X_train, y_train)

    # Predict labels for the testing set
    kernel_svm_y_pred = kernel_svm_clf.predict(X_test)

    # Calculate accuracy
    kernel_svm_accuracy = accuracy_score(y_test, kernel_svm_y_pred)
    print("Kernel SVM Accuracy: {:.2f}%".format(kernel_svm_accuracy * 100))


def k_means_clustering_classifier(X_train, y_train, X_test, y_test, class_number):

    # Create a K-means clustering model
    kmeans = KMeans(n_clusters=class_number)

    # Train the K-means model using the training data
    kmeans.fit(X_train)

    # Get the cluster labels assigned to each training data point
    train_labels = kmeans.labels_

    # Count the occurrences of each label in each cluster
    cluster_label_counts = np.zeros((class_number, class_number), dtype=int)
    for i in range(class_number):
        cluster_labels = y_train[train_labels == i]
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_label_counts[i, unique_labels] = counts

    # Assign the label to each cluster center based on the majority class
    cluster_center_labels = np.argmax(cluster_label_counts, axis=1)

    # Predict the nearest cluster center for each test data point
    test_cluster_labels = kmeans.predict(X_test)

    # Classify the test data according to the label of the nearest cluster center
    predicted_labels = cluster_center_labels[test_cluster_labels]

    # Calculate accuracy
    kmeans_accuracy = accuracy_score(y_test, predicted_labels)
    print("K-means Clustering Accuracy: {:.2f}%".format(kmeans_accuracy * 100))
