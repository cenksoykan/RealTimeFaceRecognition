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
import utils as ut
import svm

print(__doc__)

###############################################################################
# Building SVC from database

FACE_DIM = (50, 50)  # h = 50, w = 50

# Load training data from face_profiles/
FACE_PROFILE_DATA, FACE_PROFILE_NAME_INDEX, FACE_PROFILE_NAMES = ut.load_training_data(
    "../face_profiles/")

print("\n" + str(FACE_PROFILE_NAME_INDEX.shape[0]), "samples from",
      len(FACE_PROFILE_NAMES), "people are loaded")

# Build the classifier
CLF, PCA = svm.build_SVC(FACE_PROFILE_DATA, FACE_PROFILE_NAME_INDEX, FACE_DIM)

###############################################################################
# Facial Recognition In Live Tracking

DISPLAY_FACE_DIM = (200, 200)  # the displayed video stream screen dimension
SKIP_FRAME = 2  # the fixed skip frame
FRAME_SKIP_RATE = 0  # skip SKIP_FRAME frames every other frame
SCALE_FACTOR = 2  # used to resize the captured frame for face detection for faster processing speed
CASCADELOCATION = path.normpath(
    path.realpath(cv2.__file__) +
    "/../data/haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier(CASCADELOCATION)
SIDEFACE_CASCADE = cv2.CascadeClassifier(CASCADELOCATION)

if len(sys.argv) == 2:
    SCALE_FACTOR = float(sys.argv[1])
elif len(sys.argv) > 2:
    logging.error("main.py ")
# dictionary mapping used to keep track of head rotation maps
ROTATION_MAPS = {
    "left": np.array([-30, 0, 30]),
    "right": np.array([30, 0, -30]),
    "middle": np.array([0, -30, 30]),
}


def get_rotation_map(rotation):
    """Takes in an angle rotation, and returns an optimized rotation map"""
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
        break
    # resize the captured frame for face detection to increase processing speed
    RESIZED_FRAME = cv2.resize(FRAME, FRAME_SCALE)
    RESIZED_FRAME = cv2.flip(RESIZED_FRAME, 1)

    PROCESSED_FRAME = RESIZED_FRAME
    # Skip a frame if the no face was found last frame

    if FRAME_SKIP_RATE == 0:
        FACEFOUND = False
        for r in CURRENT_ROTATION_MAP:

            ROTATED_FRAME = ndimage.rotate(RESIZED_FRAME, r)

            gray = cv2.cvtColor(ROTATED_FRAME, cv2.COLOR_BGR2GRAY)

            # return tuple is empty, ndarray if detected face
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            # If frontal face detector failed, use profileface detector
            faces = faces if len(faces) else SIDEFACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            # for f in faces:
            #     x, y, w, h = [ v*SCALE_FACTOR for v in f ]
            #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
            #     cv2.putText(frame, "Prediction", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

            if len(faces):
                for f in faces:
                    # Crop out the face
                    x, y, w, h = [
                        v for v in f
                    ]  # scale the bounding box back to original frame size
                    CROPPED_FACE = ROTATED_FRAME[
                        y:y + h, x:x + w]  # img[y: y + h, x: x + w]
                    CROPPED_FACE = cv2.resize(
                        CROPPED_FACE,
                        DISPLAY_FACE_DIM,
                        interpolation=cv2.INTER_AREA)

                    # Name Prediction
                    face_to_predict = cv2.resize(
                        CROPPED_FACE, FACE_DIM, interpolation=cv2.INTER_AREA)
                    face_to_predict = cv2.cvtColor(face_to_predict,
                                                   cv2.COLOR_BGR2GRAY)
                    name_to_display = svm.predict(CLF, PCA, face_to_predict,
                                                  FACE_PROFILE_NAMES)

                    CROPPED_FACE = cv2.flip(CROPPED_FACE, 1)

                    # Display frame
                    cv2.rectangle(ROTATED_FRAME, (x, y), (x + w, y + h),
                                  (0, 255, 0))
                    cv2.putText(ROTATED_FRAME, name_to_display, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

                # rotate the frame back and trim the black paddings
                PROCESSED_FRAME = ut.trim(
                    ut.rotate_image(ROTATED_FRAME, r * (-1)), FRAME_SCALE)

                # reset the optimized rotation map
                CURRENT_ROTATION_MAP = get_rotation_map(r)
                FACEFOUND = True
                break

        if FACEFOUND:
            FRAME_SKIP_RATE = 0
            # print("Face Found")
        else:
            FRAME_SKIP_RATE = SKIP_FRAME
            # print("Face Not Found")

    else:
        FRAME_SKIP_RATE -= 1
        # print("Face Not Found")

    # print("Frame dimension: ", PROCESSED_FRAME.shape)

    cv2.putText(PROCESSED_FRAME, "Press ESC or 'q' to quit.", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    cv2.imshow("Real Time Facial Recognition", PROCESSED_FRAME)

    # get next frame
    RET, FRAME = WEBCAM.read()

WEBCAM.release()
cv2.destroyAllWindows()
