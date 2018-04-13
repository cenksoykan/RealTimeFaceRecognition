"""
====================================================
    Faces recognition and detection using OpenCV
====================================================

The dataset used is the Extended Yale Database B Cropped

  http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html

Summary:
    Used for face profile data collection in real time face training for recognition

Run:
    * Training for face recognition using the command below.
    face_profile_name is the name of the user face profile directory that you want to create in
    the default ../face_profiles/ folder for storing user face images
    and training the SVM classification model:
        python train.py [face_profile_name=<the name of the profile folder in database>]

    * Example to create a face profile named David:
        python train.py David

Usage during run time:
    press and hold 'p' to take pictures of you continuously
    once a cropped face is detected from a pop up window.
    All images are saved under ../face_profiles/face_profile_name

    press 'q' or 'ESC' to quit the application

"""

import os
import sys
import numpy as np
from scipy import ndimage
import cv2
import utils

FACE_DIM = (50, 50)
DISPLAY_FACE_DIM = (200, 200)
SKIP_FRAME = 2  # the fixed skip frame
FRAME_SKIP_RATE = 0  # skip SKIP_FRAME frames every other frame
SCALE_FACTOR = 2  # used to resize the captured frame for face detection for faster processing speed
FRONTALFACE = os.path.join(
    os.path.dirname(cv2.__file__), "data",
    "haarcascade_frontalface_default.xml")
PROFILEFACE = os.path.join(
    os.path.dirname(cv2.__file__), "data", "haarcascade_profileface.xml")
FACE_CASCADE = cv2.CascadeClassifier(FRONTALFACE)
SIDEFACE_CASCADE = cv2.CascadeClassifier(PROFILEFACE)

#  For saving face data to directory
PROFILE_FOLDER_PATH = None

# dictionary mapping used to keep track of head rotation maps
ROTATION_MAPS = {
    "left": np.array([-30, 0, 30]),
    "right": np.array([30, 0, -30]),
    "middle": np.array([0, -30, 30]),
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


if len(sys.argv) == 1:
    utils.save_data()
    exit()
elif len(sys.argv) > 2:
    print("\nError: More Than One Saving Directory Specified\n")
    exit()
else:
    PROFILE_FOLDER_PATH = utils.create_profile_in_database(sys.argv[1])

CURRENT_ROTATION_MAP = get_rotation_map(0)
WEBCAM = cv2.VideoCapture(0)
RET, FRAME = WEBCAM.read()  # get first frame
FRAME_SCALE = (int(FRAME.shape[1] / SCALE_FACTOR),
               int(FRAME.shape[0] / SCALE_FACTOR))  # (y, x)

CROPPED_FACE = []
NUM_OF_FACE_TO_COLLECT = 150
NUM_OF_FACE_SAVED = 0
UNSAVED = True

for picture in os.listdir(PROFILE_FOLDER_PATH):
    file_path = os.path.join(PROFILE_FOLDER_PATH, picture)
    if file_path.endswith(".png") or file_path.endswith(
            ".jpg") or file_path.endswith(".jpeg") or file_path.endswith(
                ".pgm"):
        NUM_OF_FACE_SAVED += 1

while RET:
    KEY = cv2.waitKey(1)
    # exit on 'q' 'esc' 'Q'
    if KEY in [27, ord('Q'), ord('q')]:
        exit()

    # resize the captured frame for face detection to increase processing speed
    RESIZED_FRAME = cv2.resize(FRAME, FRAME_SCALE)
    RESIZED_FRAME = cv2.flip(RESIZED_FRAME, 1)

    PROCESSED_FRAME = RESIZED_FRAME
    # Skip a frame if the no face was found last frame
    if FRAME_SKIP_RATE == 0:
        FACEFOUND = False
        for r in CURRENT_ROTATION_MAP:
            ROTATED_FRAME = ndimage.rotate(RESIZED_FRAME, r)
            GRAY_FRAME = cv2.cvtColor(ROTATED_FRAME, cv2.COLOR_BGR2GRAY)

            # return tuple is empty, ndarray if detected face
            faces = FACE_CASCADE.detectMultiScale(
                GRAY_FRAME,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            # If frontal face detector failed, use profileface detector
            if not len(faces):
                SIDEFACE_CASCADE.detectMultiScale(
                    GRAY_FRAME,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces):
                for f in faces:
                    x, y, w, h = [v for v in f]
                    CROPPED_FACE = GRAY_FRAME[y:y + h, x:x + w]
                    CROPPED_FACE = cv2.resize(
                        CROPPED_FACE,
                        DISPLAY_FACE_DIM,
                        interpolation=cv2.INTER_AREA)
                    CROPPED_FACE = cv2.flip(CROPPED_FACE, 1)

                    cv2.rectangle(ROTATED_FRAME, (x, y), (x + w, y + h),
                                  (0, 255, 0))
                    cv2.putText(ROTATED_FRAME, "Training Face", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

                # rotate the frame back and trim the black paddings
                PROCESSED_FRAME = utils.rotate_image(ROTATED_FRAME, r * (-1))
                PROCESSED_FRAME = utils.trim(PROCESSED_FRAME, FRAME_SCALE)

                # reset the optimized rotation map
                CURRENT_ROTATION_MAP = get_rotation_map(r)
                FACEFOUND = True
                UNSAVED = True
                break

        if FACEFOUND:
            FRAME_SKIP_RATE = 0
        else:
            FRAME_SKIP_RATE = SKIP_FRAME
    else:
        FRAME_SKIP_RATE -= 1

    cv2.putText(PROCESSED_FRAME, "Press ESC or 'q' to quit.", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

    cv2.imshow("Face Recognition", PROCESSED_FRAME)

    if len(CROPPED_FACE):
        if UNSAVED and NUM_OF_FACE_SAVED < NUM_OF_FACE_TO_COLLECT:
            cv2.imshow("Recognized Face", CROPPED_FACE)
            if KEY in [ord('P'), ord('p')]:
                FACE_TO_SAVE = cv2.resize(
                    CROPPED_FACE, FACE_DIM, interpolation=cv2.INTER_AREA)
                FACE_NAME = sys.argv[1] + "-" + str(NUM_OF_FACE_SAVED) + ".pgm"
                IMG_PATH = os.path.join(PROFILE_FOLDER_PATH, FACE_NAME)
                cv2.imwrite(IMG_PATH, FACE_TO_SAVE)
                NUM_OF_FACE_SAVED += 1
                UNSAVED = False
                print("Picture is saved:", FACE_NAME)
        else:
            exit()

    # get next frame
    RET, FRAME = WEBCAM.read()

WEBCAM.release()
cv2.destroyAllWindows()
