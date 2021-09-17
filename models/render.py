import cv2
import numpy as np

# units in cm
screen_w = 32
screen_h = 18
screen_aspect = screen_w / screen_h
camera_l = 2.299
camera_t = 0.91
screen_t = 1.719
screen_l = 0.438
screen_from_camera = [screen_t - camera_t, screen_l - camera_l]

screenW = 800
screenH = 450


def render_gaze(full_image, camera_center, cm_to_px, output):
    xScreen = output[0]
    yScreen = output[1]
    pixelGaze = [
        round(camera_center[0] - yScreen * cm_to_px),
        round(camera_center[1] + xScreen * cm_to_px)
    ]
    # render eye gaze point
    cv2.circle(full_image, (int(pixelGaze[1]), int(pixelGaze[0])), 30,
               (0, 0, 255), -1)
    return full_image


def render_gazes(img, outputs):
    full_image = np.ones((round(img.shape[0] * 2), round(img.shape[1] * 2), 3),
                         dtype=np.uint8)

    full_image_center = [
        round(full_image.shape[0] * 0.2),
        round(full_image.shape[1] * .5)
    ]
    camera_center = full_image_center

    cm_to_px = img.shape[0] * 1. / screen_h

    screen_from_camera_px = [
        round(screen_from_camera[0] * cm_to_px),
        round(screen_from_camera[1] * cm_to_px)
    ]

    screen_start = [
        camera_center[0] + screen_from_camera_px[0],
        camera_center[1] + screen_from_camera_px[1]
    ]

    full_image[screen_start[0]:screen_start[0] + img.shape[0],
               screen_start[1]:screen_start[1] +
               img.shape[1], :] = img[:, :, :]

    # render camera center
    cv2.circle(full_image, (camera_center[1], camera_center[0]), 30,
               (255, 0, 0), -1)

    for output in outputs:
        full_image = render_gaze(full_image, camera_center, cm_to_px, output)

    cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

    return full_image


def render_face_feats(img, faces, face_features):
    for i, face in enumerate(faces):
        face_image, eye_images, face_grid = face_features[i]
        cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        for eye_image in eye_images:
            cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
