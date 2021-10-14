import cv2, time
import numpy as np

cascades = '/usr/share/opencv4/haarcascades/'
face_cascade = cv2.CascadeClassifier(cascades +
                                     'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_eye.xml')


def current_time():
    return int(round(time.time() * 1000))


def get_right_left_eyes(roi_gray):
    # sort descending
    eyes = eye_cascade.detectMultiScale(roi_gray)
    eyes_sorted_by_size = sorted(eyes, key=lambda x: -x[2])
    largest_eyes = eyes_sorted_by_size[:2]
    # sort by x position
    largest_eyes.sort(key=lambda x: x[0])
    return largest_eyes


def extract_face_features(face, img, gray):
    [x, y, w, h] = face
    roi_gray = gray[y:y + h, x:x + w]
    face_image = np.copy(img[y:y + h, x:x + w])

    eyes = get_right_left_eyes(roi_gray)
    eye_images = []
    for (ex, ey, ew, eh) in eyes:
        eye_images.append(np.copy(img[y + ey:y + ey + eh, x + ex:x + ex + ew]))

    roi_color = img[y:y + h, x:x + w]

    return face_image, eye_images


def get_face_grid(face, frameW, frameH, gridSize):
    faceX, faceY, faceW, faceH = face
    return faceGridFromFaceRect(frameW, frameH, gridSize, gridSize, faceX,
                                faceY, faceW, faceH, False)


def extract_frame_features(img, grayed=False):
    start_ms = current_time()
    gray = img
    if not grayed:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detections = face_cascade.detectMultiScale(gray, 1.3, 5)

    faces = []
    face_features = []
    for [x, y, w, h] in face_detections:
        face = [x, y, w, h]
        face_image, eye_images = extract_face_features(face, img, gray)
        face_grid = get_face_grid(face, img.shape[1], img.shape[0], 25)

        faces.append(face)
        face_features.append([face_image, eye_images, face_grid])

    duration_ms = current_time() - start_ms
    print("Face and eye extraction took: ", str(duration_ms / 1000) + "s")

    return img, faces, face_features


def extract_image_features(full_img_path):
    img = cv2.imread(full_img_path)
    return extract_frame_features(img)


# This code is converted from https://github.com/CSAILVision/GazeCapture/blob/master/code/faceGridFromFaceRect.m

# Given face detection data, generate face grid data.
#
# Input Parameters:
# - frameW/H: The frame in which the detections exist
# - gridW/H: The size of the grid (typically same aspect ratio as the
#     frame, but much smaller)
# - labelFaceX/Y/W/H: The face detection (x and y are 0-based image
#     coordinates)
# - parameterized: Whether to actually output the grid or just the
#     [x y w h] of the 1s square within the gridW x gridH grid.


def faceGridFromFaceRect(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY,
                         labelFaceW, labelFaceH, parameterized):

    scaleX = gridW / frameW
    scaleY = gridH / frameH

    if parameterized:
        labelFaceGrid = np.zeros(4)
    else:
        labelFaceGrid = np.zeros(gridW * gridH)

    grid = np.zeros((gridH, gridW))

    # Use one-based image coordinates.
    xLo = round(labelFaceX * scaleX)
    yLo = round(labelFaceY * scaleY)
    w = round(labelFaceW * scaleX)
    h = round(labelFaceH * scaleY)

    if parameterized:
        labelFaceGrid = [xLo, yLo, w, h]
    else:
        xHi = xLo + w
        yHi = yLo + h

        # Clamp the values in the range.
        xLo = int(min(gridW, max(0, xLo)))
        xHi = int(min(gridW, max(0, xHi)))
        yLo = int(min(gridH, max(0, yLo)))
        yHi = int(min(gridH, max(0, yHi)))

        faceLocation = np.ones((yHi - yLo, xHi - xLo))
        grid[yLo:yHi, xLo:xHi] = faceLocation

        # Flatten the grid.
        grid = np.transpose(grid)
        labelFaceGrid = grid.flatten()

    return labelFaceGrid
