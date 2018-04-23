"""
Summary:
    SVM methods using Scikit

"""

import os
from time import time
from pickle import dump
from pickle import load

import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2

import utils

FACE_DIM = (32, 32)


def error_rate(pred, actual):
    """
    Calculate name prediction error rate

    Parameters
    ----------
    pred: ndarray (1, number_of_images_in_face_profiles)
        The predicated names of the test dataset

    actual: ndarray (1, number_of_images_in_face_profiles)
        The actual names of the test dataset

    Returns
    -------
    rate: float
        The calculated error rate

    """
    if pred.shape != actual.shape:
        return None
    rate = np.count_nonzero(pred - actual) / float(pred.shape[0])
    return rate


def build_svc(face_profile_data, face_profile_name_index, face_profile_names):
    """
    Build the SVM classification modle using the face_profile_data matrix (numOfFace X numOfPixel)
    and face_profile_name_index array, face_dim is a tuple of the dimension of each image(h,w)
    Returns the SVM classification modle

    Parameters
    ----------
    face_profile_data : ndarray (number_of_images_in_face_profiles, width * height of the image)
        The pca that contains the top eigenvectors extracted
        using approximated Singular Value Decomposition of the data

    face_profile_name_index : ndarray
        The name corresponding to the face profile is encoded in its index

    Returns
    -------
    clf : theano object
        The trained SVM classification model

    pca : theano ojbect
        The PCA that contains the top 128 eigenvectors extracted
        using approximated Singular Value Decomposition of the data

    """
    x = face_profile_data
    y = face_profile_name_index

    n_samples = y.shape[0]
    n_features = x.shape[1]
    n_classes = len(face_profile_names)

    print("\n%d samples from %d people are loaded\n" % (n_samples, n_classes))
    print("Samples:", n_samples)
    print("Features:", n_features)
    print("Classes:", n_classes)

    # Split into a training set and a test set using a stratified k fold
    # split into a training and testing set

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.25, random_state=42)

    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 128  # maximum number of components to keep

    print("\nExtracting the top %d eigenfaces from %d faces" %
          (n_components, x_train.shape[0]))

    t_0 = time()
    pca = PCA(
        n_components=n_components, svd_solver='randomized',
        whiten=True).fit(x_train)
    print("done in %.3fs" % (time() - t_0))

    # eigenfaces = pca.components_.reshape((n_components, FACE_DIM[0],
    #                                       FACE_DIM[1]))

    # This portion of the code is used if the data is scarce, it uses the number
    # of imputs as the number of features
    # pca = PCA(n_components=None, whiten=True).fit(x_train)
    # eigenfaces = pca.components_.reshape((pca.components_.shape[0], FACE_DIM[0], FACE_DIM[1]))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t_0 = time()
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    print("done in %.3fs" % (time() - t_0))

    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t_0 = time()
    param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.1],
    }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

    # clf = GridSearchCV(
    #     SVC(kernel='rbf',
    #         class_weight='balanced',
    #         cache_size=200,
    #         coef0=0.0,
    #         decision_function_shape='ovr',
    #         degree=3,
    #         max_iter=-1,
    #         probability=False,
    #         random_state=None,
    #         shrinking=True,
    #         tol=0.001,
    #         verbose=False), param_grid)
    clf = clf.fit(x_train_pca, y_train)

    print("done in %.3fs" % (time() - t_0))
    print("Best estimator found by grid search:")
    # print(clf.best_estimator_)

    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("\nPredicting people's names on the test set")
    t_0 = time()
    y_pred = clf.predict(x_test_pca)
    print("\nPrediction took %.8fs per sample on average" %
          ((time() - t_0) / float(y_pred.shape[0])))

    # print(
    #     classification_report(y_test, y_pred, target_names=face_profile_names))
    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    rate = error_rate(y_pred, y_test)
    print("\nTest Error Rate:\t %.4f%%" % (rate * 100))
    print("Test Recognition Rate:\t%.4f%%" % ((1.0 - rate) * 100))

    def plot_gallery(images, titles, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.5 * n_col, 2 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape(FACE_DIM), cmap='gray')
            plt.title(titles[i], size=8)
            plt.xticks(())
            plt.yticks(())

    def title(y_pred, y_test, target_names, i):
        """Helper function to plot the result of the prediction"""
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return 'pred: %s\ntest: %s' % (pred_name, true_name)

    prediction_titles = [
        title(y_pred, y_test, face_profile_names, i)
        for i in range(y_pred.shape[0])
    ]

    plot_gallery(x_test, prediction_titles)

    plt.show()

    return clf, pca, face_profile_names


def fetch_data():
    """
    Saves and returns image training data

    """
    # Load training data from face_profiles/
    face_profile_data, face_profile_name_index, face_profile_names = utils.load_training_data(
    )

    # Build the classifier
    face_profile = build_svc(face_profile_data, face_profile_name_index,
                             face_profile_names)

    data_dir = os.path.join(os.path.dirname(__file__), "../temp")
    data_path = os.path.join(data_dir, "SVM.pkl")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Save the classifier
    with open(data_path, 'wb') as file:
        dump(face_profile, file)

    print("\nTraining data is successfully saved\n")
    return face_profile


def predict(face):
    """
    Predict the name of the supplied image from the list of face profile names

    Parameters
    ----------

    img: ndarray
        The input image for prediction

    Returns
    -------
    name : string
        The predicated name

    """
    # Building SVC from database

    data_path = os.path.join(os.path.dirname(__file__), "../temp", "SVM.pkl")

    if os.path.exists(data_path):
        with open(data_path, 'rb') as file:
            clf, pca, face_profile_names = load(file)
    else:
        clf, pca, face_profile_names = fetch_data()

    img = cv2.resize(face, FACE_DIM, interpolation=cv2.INTER_AREA)
    img = cv2.convertScaleAbs(img)
    img = img.ravel()
    # Apply dimensionality reduction on img, img is projected on the first principal components
    # previous extracted from the Yale Extended dataset B.
    principle_components = pca.transform(np.array(img).reshape(1, -1))
    pred = clf.predict(principle_components)
    name = face_profile_names[np.int(pred)]
    return name
