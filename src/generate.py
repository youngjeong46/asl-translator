"""
Pic module that writes image files using OpenCV
"""

import os
import cv2
import numpy as np


UPPER_LEFT = (100, 100)
BOTTOM_RIGHT = (500, 500)
SAVE_PATH = "../data/raw/normal/"
SAVE_KNN = "../data/raw/knn/"


def make_pics():
    """
    Produces ASL letter images for creating a image dataset
    creates both regular image and Background Subtracter image (KNN)
    """
    knn_sub = cv2.createBackgroundSubtractorKNN()
    camera = cv2.VideoCapture(0)
    num = 0

    while camera.isOpened():
        _, frame = camera.read()
        cv2.rectangle(frame, UPPER_LEFT, BOTTOM_RIGHT,
                      (0, 255, 0), 1)  # green box
        rect_img = frame[UPPER_LEFT[1]:BOTTOM_RIGHT[1],
                         UPPER_LEFT[0]: BOTTOM_RIGHT[0]]
        fg_KNN = knn_sub.apply(rect_img)
        fg_KNN = cv2.cvtColor(fg_KNN, cv2.COLOR_GRAY2BGR)
        final = np.hstack((rect_img, fg_KNN))

        cv2.imshow('final', final)

        key = cv2.waitKey(10)
        if key == 27:
            break

        if key in list(range(97, 123)) + [49, 50, 51]:
            # 97-122 for a-z, 1 for nothing, 2 for space, 3 for del
            if not os.path.exists(SAVE_PATH):
                os.mkdir(SAVE_PATH)

            if not os.path.exists(SAVE_KNN):
                os.mkdir(SAVE_KNN)

            dir_Name = SAVE_PATH + chr(key).upper() + "/"
            dir_Name_KNN = SAVE_KNN + chr(key).upper() + "/"

            if not os.path.exists(dir_Name):
                os.mkdir(dir_Name)
            if not os.path.exists(dir_Name_KNN):
                os.mkdir(dir_Name_KNN)

            cv2.imwrite(dir_Name + f"{num}.jpg", rect_img)
            cv2.imwrite(dir_Name_KNN+f"{num}.jpg", fg_KNN)
            num += 1

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    make_pics()
