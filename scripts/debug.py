"""
Debug SVM prediction

"""

from os import path

import cv2

import utils
from svm import fetch_data
from svm import predict

DATA_PATH = path.join(
    path.dirname(__file__), "../face_profiles", "debug.test", "debug.test.pgm")

if not path.exists(DATA_PATH):
    print("\nError: There is no picture in this direction\n")
    exit()

if utils.check_image_format(DATA_PATH):
    FACE = cv2.imread(DATA_PATH, 0)
else:
    print(
        "\nError: File extension has to be one of these: png, jpg, jpeg, pgm\n"
    )
    exit()

fetch_data()

PREDICTION_NAME = predict(FACE)
print("This is picture of", "\"" + PREDICTION_NAME + "\"")
exit()
