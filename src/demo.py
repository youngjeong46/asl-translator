import argparse
import pickle
import cv2
import numpy as np
# import time


from keras.models import load_model

ap = argparse.ArgumentParser(
    description="Create American Sign Language translation demo video using Deep Learning architectures.")

ap.add_argument("-m", "--model", type=str, required=True,
                help="Name of pre-trained deep learning architecture to use. Options are: vgg16, mobilenetv1")
ap.add_argument("-o", "--output", required=True,
                help="path to output video file")
ap.add_argument("-f", "--fps", type=int, default=20,
                help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
                help="codec of output video")
ARGS = vars(ap.parse_args())

MODELS = {
    "vgg16": '../models/vgg16.h5',
    "mobilenetv1": '../models/mobilenet_v1.h5',
    "mobilenetv2": '../models/mobilenet_v2.h5'
}

CHARACTER = pickle.load(open("../models/label_map.pickle", "rb"))
MODEL_NAME = ARGS["model"]

if MODEL_NAME not in MODELS.keys():
    raise AssertionError(
        "The --model command line argument should be vgg16 or mobilenetv1")

if MODEL_NAME == "vgg16":
    DIMS = (200, 200)
else:
    DIMS = (224, 224)

print("Loading "+MODEL_NAME+" architecture...")
MODEL = load_model(MODELS[MODEL_NAME])
UPPER_LEFT = (800, 100)
BOTTOM_RIGHT = (1200, 500)
WORD = ""


def mapping_next_letter(curr, nextChar):
    """
    Helper function to map the letter using pre-made dict
    """
    if nextChar == "space":
        curr += " "
    elif nextChar == "del":
        curr = curr[:-1]
    elif nextChar == "nothing":
        curr = curr
    else:
        curr += nextChar
    return curr


def predict_image(image):
    """
    Predicts the image using the imported DL model
    """
    image = np.array(image, dtype='float32')
    image = np.flip(image, axis=2)
    image[:, :, 2] -= 123.68
    image[:, :, 1] -= 116.779
    image[:, :, 0] -= 103.939
    image /= 255

    if MODEL_NAME == "vgg16":
        image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    pred_array = MODEL.predict(image)

    idx = np.argmax(pred_array)
    result = CHARACTER[idx]

    return result


camera = cv2.VideoCapture(0)
# start = time.time()
# num_frames = 0
fourcc = cv2.VideoWriter_fourcc(*ARGS["codec"])
writer = None
(h, w) = (None, None)
countFrame = 0

while camera.isOpened():
    ret, frame = camera.read()  # read and capture frames
    frame = cv2.flip(frame, 1)  # flip image
    cv2.rectangle(frame, UPPER_LEFT, BOTTOM_RIGHT,
                  (0, 255, 0), 1)  # green box
    rect_img = frame[UPPER_LEFT[1]:BOTTOM_RIGHT[1],
                     UPPER_LEFT[0]: BOTTOM_RIGHT[0]]
    target = cv2.resize(rect_img, DIMS)
    prediction = predict_image(target)

    cv2.putText(frame, mapping_next_letter(WORD, prediction), (800, 600),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (5, 5, 5), 2)

    if writer is None:
        (h, w) = frame.shape[:2]
        writer = cv2.VideoWriter(ARGS["output"], fourcc, ARGS["fps"],
                                 (w, h), True)

    writer.write(frame)

    cv2.imshow('video output', frame)

    k = cv2.waitKey(1)
    if k == 32:
        WORD = mapping_next_letter(WORD, prediction)
    if k == 27:
        break
    countFrame += 1
    # num_frames += 1

# print(f"frames elapsed: {num_frames}")
# end = time.time()
# print(f"{end - start} seconds elapsed.")
print(countFrame)
writer.release()
camera.release()
cv2.destroyAllWindows()
