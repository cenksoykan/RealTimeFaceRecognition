"""
====================================================
    Faces recognition and detection using OpenCV
====================================================

The dataset used is the Extended Yale Database B Cropped

  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html

Summary:
    Real time facial tracking and recognition using harrcascade and SVM

To Run:
    * To run it without options
        python main.py

    * Or running with options (By default, scale_multiplier = 4):
        python main.py [scale_multiplier=<full screensize divided by scale_multiplier>]

    * Say you want to run with 1/2 of the full screen size, specify that scale_multiplier = 4:
        python main.py 4

Usage:
    press 'q' or 'ESC' to quit the application

"""

from os import path
import sys
import logging
from scipy import ndimage
import numpy as np
import cv2
import utils
from svm import predict

print(__doc__)

###############################################################################
# Facial Recognition In Live Tracking

FACE_DIM = (50, 50)  # h = 50, w = 50
DISPLAY_FACE_DIM = (200, 200)  # the displayed video stream screen dimension
SKIP_FRAME = 2  # the fixed skip frame
FRAME_SKIP_RATE = 0  # skip SKIP_FRAME frames every other frame
SCALE_FACTOR = 2  # used to resize the captured frame for face detection for faster processing speed
FRONTALFACE = path.join(
    path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
PROFILEFACE = path.join(
    path.dirname(cv2.__file__), "data", "haarcascade_profileface.xml")
FACE_CASCADE = cv2.CascadeClassifier(FRONTALFACE)
SIDEFACE_CASCADE = cv2.CascadeClassifier(PROFILEFACE)

if len(sys.argv) == 2:
    SCALE_FACTOR = float(sys.argv[1])
elif len(sys.argv) > 2:
    logging.error("main.py ")

# dictionary mapping used to keep track of head rotation maps
ROTATION_MAPS = {
    "middle": np.array([0, -30, 30]),
    "left": np.array([-30, 0, 30]),
    "right": np.array([30, 0, -30]),
}


def get_rotation_map(rotation):
    """
    Takes in an angle rotation, and returns an optimized rotation map

    """
    if rotation > 0:
        return ROTATION_MAPS.get("right", None)
    elif rotation < 0:
        return ROTATION_MAPS.get("left", None)
    return ROTATION_MAPS.get("middle", None)


CURRENT_ROTATION_MAP = get_rotation_map(0)
WEBCAM = cv2.VideoCapture(0)
RET, FRAME = WEBCAM.read()  # get first frame
FRAME_SCALE = (int(FRAME.shape[1] / SCALE_FACTOR),
               int(FRAME.shape[0] / SCALE_FACTOR))  # (y, x)

CROPPED_FACE = []

while RET:
    KEY = cv2.waitKey(1)
    # exit on 'q' 'esc' 'Q'
    if KEY in [27, ord('Q'), ord('q')]:
        exit()

    # resize the captured frame for face detection to increase processing speed
    RESIZED_FRAME = cv2.resize(
        FRAME, FRAME_SCALE, interpolation=cv2.INTER_AREA)
    RESIZED_FRAME = cv2.flip(RESIZED_FRAME, 1)

    PROCESSED_FRAME = RESIZED_FRAME
    # Skip a frame if the no face was found last frame
    if FRAME_SKIP_RATE == 0:
        FACEFOUND = False
        for r in CURRENT_ROTATION_MAP:
            ROTATED_FRAME = ndimage.rotate(RESIZED_FRAME, r)
            GRAY_FRAME = cv2.cvtColor(ROTATED_FRAME, cv2.COLOR_BGR2GRAY)
            GRAY_FRAME = cv2.convertScaleAbs(GRAY_FRAME)

            # return tuple is empty, ndarray if detected face
            faces = FACE_CASCADE.detectMultiScale(
                GRAY_FRAME,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            # If frontal face detector failed, use profileface detector
            if not len(faces):
                faces = SIDEFACE_CASCADE.detectMultiScale(
                    GRAY_FRAME,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces):
                for f in faces:
                    # Crop out the face
                    x, y, w, h = [v for v in f]
                    CROPPED_FACE = GRAY_FRAME[y:y + h, x:x + w]
                    CROPPED_FACE = cv2.resize(
                        CROPPED_FACE, FACE_DIM, interpolation=cv2.INTER_AREA)
                    CROPPED_FACE = cv2.flip(CROPPED_FACE, 1)

                    name_to_display = predict(CROPPED_FACE)

                    cv2.rectangle(ROTATED_FRAME, (x, y), (x + w, y + h),
                                  (0, 255, 0))
                    cv2.putText(ROTATED_FRAME, name_to_display, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

                # rotate the frame back and trim the black paddings
                PROCESSED_FRAME = utils.rotate_image(ROTATED_FRAME, r * (-1))
                PROCESSED_FRAME = utils.trim(PROCESSED_FRAME, FRAME_SCALE)

                # reset the optimized rotation map
                CURRENT_ROTATION_MAP = get_rotation_map(r)
                FACEFOUND = True

        if FACEFOUND:
            FRAME_SKIP_RATE = 0
        else:
            FRAME_SKIP_RATE = SKIP_FRAME
    else:
        FRAME_SKIP_RATE -= 1

    # print("Frame dimension: ", PROCESSED_FRAME.shape)

    cv2.putText(PROCESSED_FRAME, "Press ESC or 'q' to quit.", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    cv2.imshow("Face Recognition", PROCESSED_FRAME)

    # get next frame
    RET, FRAME = WEBCAM.read()

WEBCAM.release()
cv2.destroyAllWindows()
