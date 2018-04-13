"""
========================
    Face recognition
========================

Summary:
    Face recognition using harrcascade and SVM

To Run:
    * To run it specify that the file you want to predict result
        python pred.py somePicture.pgm

"""

from os import path
import sys
import cv2
from svm import predict
from utils import check_image_format

print(__doc__)

FACE_DIM = (50, 50)  # h = 50, w = 50
DISPLAY_FACE_DIM = (200, 200)
FRONTALFACE = path.join(
    path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
PROFILEFACE = path.join(
    path.dirname(cv2.__file__), "data", "haarcascade_profileface.xml")
FACE_CASCADE = cv2.CascadeClassifier(FRONTALFACE)
SIDEFACE_CASCADE = cv2.CascadeClassifier(PROFILEFACE)

if len(sys.argv) == 1:
    print("\nError: No Any Face Picture Specified\n")
    exit()
elif len(sys.argv) > 2:
    print("\nError: More Than One Face Picture Specified\n")
    exit()
else:
    DATA_PATH = sys.argv[1]

if not path.exists(DATA_PATH):
    print("\nError: There is no picture in this direction\n")
    exit()

if check_image_format(DATA_PATH):
    FACE = cv2.imread(DATA_PATH, 0)
else:
    print(
        "\nError: File extension has to be one of these: png, jpg, jpeg, pgm\n"
    )
    exit()

print("This is picture of", "\"" + predict(FACE) + "\"")
