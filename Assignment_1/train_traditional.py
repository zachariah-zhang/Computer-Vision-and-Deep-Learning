import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
from classifer_tranditional import svm_classifier, kernel_svm_classifier, k_means_clustering_classifier
from feature_extractor import extract_features
import pickle
import os


def train():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Intel Image Classification')
    parser.add_argument('--feature', type=str, choices=['hog', 'sift'], default='hog',
                        help='Feature extractor to use (default: hog)')
    parser.add_argument('--classifier', type=str, choices=['svm', 'kernel_svm', 'kmeans'], default='svm',
                        help='Classifier to use (default: svm)')
    args = parser.parse_args()

    # extract features
    if args.feature == 'sift':
        with open("./features/sift/X_train_sift.pkl", "rb") as f:
            X_train = pickle.load(f)
        with open("./features/sift/y_train_sift.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open("./features/sift/X_test_sift.pkl", "rb") as f:
            X_test = pickle.load(f)
        with open("./features/sift/y_test_sift.pkl", "rb") as f:
            y_test = pickle.load(f)

    # extract features
    if args.feature == 'hog':
        with open('./features/hog/X_train_hog.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open("./features/hog/y_train_hog.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open("./features/hog/X_test_hog.pkl", "rb") as f:
            X_test = pickle.load(f)
        with open("./features/hog/y_test_hog.pkl", "rb") as f:
            y_test = pickle.load(f)

    # train and evaluate the classifier
    if args.classifier == 'svm':
        svm_classifier(X_train, y_train, X_test, y_test)
    elif args.classifier == 'kernel_svm':
        kernel_svm_classifier(X_train, y_train, X_test, y_test)
    elif args.classifier == 'kmeans':
        k_means_clustering_classifier(
            X_train, y_train, X_test, y_test, 6)


if __name__ == '__main__':
    train()
