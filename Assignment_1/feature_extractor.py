import torch
import torchvision
from torchvision import transforms, datasets
from sklearn import svm
import cv2
import numpy as np
from skimage.feature import hog
import pickle


def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, transform_sqrt=True)
    return np.array(features)


def extract_sift_features(image):
    if image.dtype != 'uint8':
        image = cv2.convertScaleAbs(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return keypoints, np.array(descriptors)


def extract_features(data_loader, feature_extractor):
    features = []
    labels = []
    i = 0
    for image, target in data_loader:
        # Convert the tensor image to numpy array (H, W, C)
        image = image.permute(1, 2, 0).numpy()
        if feature_extractor == 'sift':
            keypoints, descriptors = extract_sift_features(image)
            features.append(descriptors)
            labels.append(target)
        elif feature_extractor == 'hog':
            hog_features = extract_hog_features(image)
            features.append(hog_features)
            labels.append(target)
        i += 1
        if (i + 1) % 100 == 0:
            print("Fraction of completed iterations:", (i + 1)/len(data_loader))

    return features, labels


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    training_dataset = torchvision.datasets.ImageFolder(
        './Data/seg_train', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(
        './Data/seg_test', transform=transform)
    X_train_sift, y_train_sift = extract_features(training_dataset, 'sift')
    X_test_sift, y_test_sift = extract_features(test_dataset, 'sift')
    X_train_hog, y_train_hog = extract_features(training_dataset, 'hog')
    X_test_hog, y_test_hog = extract_features(test_dataset, 'hog')
    with open("./features/sift/X_train_sift.pkl", "wb") as f:
        pickle.dump(X_train_sift, f)
    f.close()
    with open("./features/sift/X_test_sift.pkl", "wb") as f:
        pickle.dump(X_test_sift, f)
    f.close()
    with open("./features/sift/y_train_sift.pkl", "wb") as f:
        pickle.dump(y_train_sift, f)
    f.close()
    with open("./features/sift/y_test_sift.pkl", "wb") as f:
        pickle.dump(y_test_sift, f)
    f.close()
    with open("./features/hog/X_test_hog.pkl", "wb") as f:
        pickle.dump(X_test_hog, f)
    f.close()
    with open("./features/hog/X_train_hog.pkl", "wb") as f:
        pickle.dump(X_train_hog, f)
    f.close()
    with open("./features/hog/y_train_hog.pkl", "wb") as f:
        pickle.dump(y_train_hog, f)
    f.close()
    with open("./features/hog/y_test_hog.pkl", "wb") as f:
        pickle.dump(y_test_hog, f)
    f.close()
