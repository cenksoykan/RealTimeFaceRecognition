"""
========================
    Face recognition
========================

Summary:
    Real time facial tracking and recognition using harrcascade and SVM

To Run:
    * To run it without options
        python main.py

    * To run it specify that the file you want to predict result
        python main.py somePicture.pgm

Usage:
    press 'q' or 'ESC' to quit the application

"""

from os import path
import sys
from scipy import ndimage
import cv2
import utils
from svm import predict

print(__doc__)

if len(sys.argv) == 2:
    DATA_PATH = sys.argv[1]

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
    print("This is picture of", "\"" + predict(FACE) + "\"")
    exit()
elif len(sys.argv) > 2:
    print("\nError: Specify only one picture at a time\n")
    exit()

FACE_DIM = (50, 50)  # h = 50, w = 50
DISPLAY_FACE_DIM = (200, 200)  # the displayed video stream screen dimension
SKIP_FRAME = 2  # the fixed skip frame
FRAME_SKIP_RATE = 0  # skip SKIP_FRAME frames every other frame
SCALE_FACTOR = 2  # used to resize the captured frame for face detection for faster processing speed
CURRENT_ROTATION_MAP = utils.get_rotation_map(0)
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

            faces = utils.detect_faces(GRAY_FRAME)

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
                CURRENT_ROTATION_MAP = utils.get_rotation_map(r)
                FACEFOUND = True

        if FACEFOUND:
            FRAME_SKIP_RATE = 0
        else:
            FRAME_SKIP_RATE = SKIP_FRAME
    else:
        FRAME_SKIP_RATE -= 1

    cv2.putText(PROCESSED_FRAME, "Press ESC or 'q' to quit.", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    cv2.imshow("Face Recognition", PROCESSED_FRAME)

    # get next frame
    RET, FRAME = WEBCAM.read()

WEBCAM.release()
cv2.destroyAllWindows()
