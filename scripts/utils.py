"""
This file is part of Cogs 109 Project.

Summary: Utilties used for facial tracking in OpenCV

"""

import os
import logging
from shutil import rmtree
from errno import EEXIST

import numpy as np
import cv2

FACE_DIM = (50, 50)
FRONTALFACE = os.path.join(
    os.path.dirname(cv2.__file__), "data",
    "haarcascade_frontalface_default.xml")
PROFILEFACE = os.path.join(
    os.path.dirname(cv2.__file__), "data", "haarcascade_profileface.xml")
FACE_CASCADE = cv2.CascadeClassifier(FRONTALFACE)
SIDEFACE_CASCADE = cv2.CascadeClassifier(PROFILEFACE)

ROTATION_MAPS = {
    "middle": np.array([0, -30, 30]),
    "left": np.array([-30, 0, 30]),
    "right": np.array([30, 0, -30]),
}


def check_image_format(img: str):
    """
    Check if image format is one of these: png, jpg, jpeg, pgm

    """
    extensions = img.endswith(".png") or img.endswith(".jpg") or img.endswith(
        ".jpeg") or img.endswith(".pgm")
    return bool(extensions)


def read_face_profile(face_profile: str, face_profile_name_index: int):
    """
    Reads all the images from one specified face profile into ndarrays

    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile

    face_profile_name_index: int
        The name corresponding to the face profile is encoded in its index

    dim: tuple = (int, int)
        The new dimensions of the images to resize to

    Returns
    -------
    x_data : numpy array, shape = (number_of_faces_in_one_profile, pixel_width * pixel_height)
        A face data array contains the face image pixel rgb values of all the images
        in the specified face profile

    y_data : numpy array, shape = (number_of_images_in_face_profiles, 1)
        A face_profile_index data array contains the index of the face profile name
        in the specified face profile directory

    """
    x_data = np.array([])
    index = 0
    for the_file in os.listdir(face_profile):
        file_path = os.path.join(face_profile, the_file)
        if check_image_format(file_path):
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, FACE_DIM, interpolation=cv2.INTER_AREA)
            img = cv2.convertScaleAbs(img)
            img = img.ravel()
            x_data = img if not x_data.shape[0] else np.vstack((x_data, img))
            index += 1

    y_data = np.empty(index, dtype=int)
    y_data.fill(face_profile_name_index)
    return x_data, y_data


def clean_profile(face_profile_directory: str):
    """
    Deletes empty face profiles in face profile directory
    and logs error if face profiles contain very few images

    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory

    """
    img_min = 10
    profile_directory_list = os.listdir(face_profile_directory)
    for face_profile in profile_directory_list:
        if "." not in str(face_profile):
            profile_path = os.path.join(face_profile_directory, face_profile)
            index = 0
            for the_file in os.listdir(profile_path):
                file_path = os.path.join(profile_path, the_file)
                if check_image_format(file_path):
                    index += 1
            if index == 0:
                rmtree(profile_path)
                logging.warning(
                    "\nDeleted \"%s\" because it contains no images",
                    face_profile)
            if index < img_min:
                logging.warning(
                    "\nProfile \"%s\" contains very few images (At least %d images are needed)\n",
                    face_profile, img_min)
                profile_directory_list.remove(face_profile)
    return profile_directory_list


def load_training_data():
    """
    Loads all the images from the face profile directory into ndarrays

    Returns
    -------
    x_data : numpy array, shape = (number_of_faces_in_profiles, pixel_width * pixel_height)
        A face data array contains the face image pixel rgb values of all face_profiles

    y_data : numpy array, shape = (number_of_face_profiles, 1)
        A face_profile_index data array contains the indexes of all the face profile names

    """
    face_profile_directory = os.path.join(
        os.path.dirname(__file__), "../face_profiles/")

    # delete profile directory without images
    profile_directory_list = clean_profile(face_profile_directory)

    # Get a the list of folder names in face_profile as the profile names
    face_profile_names = [
        d for d in profile_directory_list if "." not in str(d)
    ]

    if len(face_profile_names) < 2:
        logging.error(
            "\nFace profile contains very few profiles (At least 2 profiles are needed)"
        )
        exit()

    first_data_name = str(face_profile_names[0])
    first_data_path = os.path.join(face_profile_directory, first_data_name)
    x_data, y_data = read_face_profile(first_data_path, 0)
    print("Loading Database:")
    print(1, "\t->", x_data.shape[0], "images are loaded from",
          "\"" + first_data_name + "\"")
    for i in range(1, len(face_profile_names)):
        directory_name = str(face_profile_names[i])
        directory_path = os.path.join(face_profile_directory, directory_name)
        temp_x, temp_y = read_face_profile(directory_path, i)
        x_data = np.concatenate((x_data, temp_x), axis=0)
        y_data = np.append(y_data, temp_y)
        print(i + 1, "\t->", temp_x.shape[0], "images are loaded from",
              "\"" + directory_name + "\"")
    return x_data, y_data, face_profile_names


def rotate_image(image: str, rotation: int, scale: float = 1.0):
    """
    Rotate an image rgb matrix with the same dimensions

    Parameters
    ----------
    image: string
        the image rgb matrix

    rotation: int
        The rotation angle in which the image rotates to

    scale: float
        The scale multiplier of the rotated image

    Returns
    -------
    rot_img : numpy array
        Rotated image after rotation

    """
    if rotation == 0:
        return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, scale)
    rot_img = cv2.warpAffine(
        image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return rot_img


def trim(image: str, dim: tuple):
    """
    Trim the four sides(black paddings) of the image matrix
    and crop out the middle with a new dimension

    Parameters
    ----------
    image: string
        the image rgb matrix

    dim: tuple (int, int)
        The new dimen the image is trimmed to

    Returns
    -------
    trimmed_img : numpy array
        The trimmed image after removing black paddings from four sides

    """

    # if the img has a smaller dimension then return the origin image
    if dim[1] >= image.shape[0] and dim[0] >= image.shape[1]:
        return image
    x = int((image.shape[0] - dim[1]) / 2) + 1
    y = int((image.shape[1] - dim[0]) / 2) + 1
    trimmed_img = image[x:x + dim[1], y:y + dim[0]]  # crop the image
    return trimmed_img


def clean_directory(face_profile: str):
    """
    Deletes all the files in the specified face profile

    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile

    """

    for the_file in os.listdir(face_profile):
        file_path = os.path.join(face_profile, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            rmtree(file_path)


def create_directory(face_profile: str):
    """
    Create a face profile directory for saving images

    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile

    """
    try:
        print("Making directory")
        os.makedirs(face_profile)
    except OSError as exception:
        if exception.errno != EEXIST:
            print(
                "The specified face profile already existed, it will be override"
            )
            raise


def create_profile_in_database(face_profile_name: str,
                               database_path: str = "../face_profiles/",
                               clean_dir: bool = False):
    """
    Create a face profile directory in the database

    Parameters
    ----------
    face_profile_name: string
        The specified face profile name of a specified face profile folder

    database_path: string
        Default database directory

    clean_dir: boolean
        Clean the directory if the user already exists

    Returns
    -------
    face_profile_path: string
        The path of the face profile created

    """
    face_profile_path = os.path.join(
        os.path.dirname(__file__), database_path, face_profile_name)
    create_directory(face_profile_path)
    # Delete all the pictures before recording new
    if clean_dir:
        clean_directory(face_profile_path)
    return face_profile_path


def get_rotation_map(rotation: int):
    """
    Takes in an angle rotation, and returns an optimized rotation map

    """
    if rotation > 0:
        return ROTATION_MAPS.get("right", None)
    elif rotation < 0:
        return ROTATION_MAPS.get("left", None)
    return ROTATION_MAPS.get("middle", None)


def detect_faces(frame):
    """
    Detect faces in frame

    """
    # return tuple is empty, ndarray if detected face
    faces = FACE_CASCADE.detectMultiScale(
        frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # If frontal face detector failed, use profileface detector
    if not len(faces):
        faces = SIDEFACE_CASCADE.detectMultiScale(
            frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

    return faces
