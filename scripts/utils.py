"""
This file is part of Cogs 109 Project.

Summary: Utilties used for facial tracking in OpenCV

"""

import os
from pickle import dump
from errno import EEXIST
import logging
import shutil
import numpy as np
import cv2

###############################################################################
# Used For Facial Tracking and Traning in OpenCV


def read_images_from_single_face_profile(face_profile,
                                         face_profile_name_index,
                                         dim=(50, 50)):
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
    x_data : numpy array, shape = (number_of_faces_in_one_face_profile, face_pixel_width * face_pixel_height)
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
        if file_path.endswith(".png") or file_path.endswith(
                ".jpg") or file_path.endswith(".jpeg") or file_path.endswith(
                    ".pgm"):
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img = cv2.convertScaleAbs(img)
            img_data = img.ravel()
            x_data = img_data if not x_data.shape[0] else np.vstack((x_data,
                                                                     img_data))
            index += 1

    y_data = np.empty(index, dtype=int)
    y_data.fill(face_profile_name_index)
    return x_data, y_data


def clean_profile(face_profile_directory):
    """
    Deletes empty face profiles in face profile directory
    and logs error if face profiles contain very few images

    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory

    """
    profile_directory_list = os.listdir(face_profile_directory)
    for face_profile in profile_directory_list:
        if "." not in str(face_profile):
            profile_path = os.path.join(face_profile_directory, face_profile)
            index = 0
            for the_file in os.listdir(profile_path):
                file_path = os.path.join(profile_path, the_file)
                if file_path.endswith(".png") or file_path.endswith(
                        ".jpg") or file_path.endswith(
                            ".jpeg") or file_path.endswith(".pgm"):
                    index += 1
            if index == 0:
                shutil.rmtree(profile_path)
                print("\nDeleted", profile_path,
                      "because it contains no images")
            if index < 2:
                logging.warning(
                    "\nFace profile \"" + str(profile_path) +
                    "\" contains very few images (At least 2 images are needed)\n"
                )
                profile_directory_list.remove(face_profile)
    return profile_directory_list


def load_training_data():
    """
    Loads all the images from the face profile directory into ndarrays

    Returns
    -------
    x_data : numpy array, shape = (number_of_faces_in_face_profiles, face_pixel_width * face_pixel_height)
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
    x_data, y_data = read_images_from_single_face_profile(first_data_path, 0)
    print("Loading Database:")
    print(1, "\t->", x_data.shape[0], "images are loaded from",
          "\"" + first_data_name + "\"")
    for i in range(1, len(face_profile_names)):
        directory_name = str(face_profile_names[i])
        directory_path = os.path.join(face_profile_directory, directory_name)
        temp_x, temp_y = read_images_from_single_face_profile(
            directory_path, i)
        x_data = np.concatenate((x_data, temp_x), axis=0)
        y_data = np.append(y_data, temp_y)
        print(i + 1, "\t->", temp_x.shape[0], "images are loaded from",
              "\"" + directory_name + "\"")
    print("\n" + str(y_data.shape[0]), "samples from", len(face_profile_names),
          "people are loaded")
    return x_data, y_data, face_profile_names


def rotate_image(img, rotation, scale=1.0):
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
        return img
    h, w = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, scale)
    rot_img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    return rot_img


def trim(img, dim):
    """
    Trim the four sides(black paddings) of the image matrix
    and crop out the middle with a new dimension

    Parameters
    ----------
    img: string
        the image rgb matrix

    dim: tuple (int, int)
        The new dimen the image is trimmed to

    Returns
    -------
    trimmed_img : numpy array
        The trimmed image after removing black paddings from four sides

    """

    # if the img has a smaller dimension then return the origin image
    if dim[1] >= img.shape[0] and dim[0] >= img.shape[1]:
        return img
    x = int((img.shape[0] - dim[1]) / 2) + 1
    y = int((img.shape[1] - dim[0]) / 2) + 1
    trimmed_img = img[x:x + dim[1], y:y + dim[0]]  # crop the image
    return trimmed_img


def clean_directory(face_profile):
    """
    Deletes all the files in the specified face profile

    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile

    """

    for the_file in os.listdir(face_profile):
        file_path = os.path.join(face_profile, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def create_directory(face_profile):
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


def create_profile_in_database(face_profile_name,
                               database_path="../face_profiles/",
                               clean_dir=False):
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


def save_data():
    """
    Saves image training data

    """
    from svm import build_svc as svc
    # Load training data from face_profiles/
    face_profile_data, face_profile_name_index, face_profile_names = load_training_data(
    )

    # Build the classifier
    face_profile = svc(face_profile_data, face_profile_name_index,
                       face_profile_names)

    data_dir = os.path.join(os.path.dirname(__file__), "../temp")
    data_path = os.path.join(data_dir, "SVM.pkl")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Save the classifier
    with open(data_path, 'wb') as f:
        dump(face_profile, f)

    print("\nTraining data is saved\n")
    return face_profile
