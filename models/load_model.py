import numpy as np

# import warnings
# warnings.filterwarnings('ignore')

# suprress Caffe verbose prints
import os
os.environ['GLOG_minloglevel'] = '2' 

import caffe
import cv2

import scipy.io
import sys
import os

from .render import *
from .face_detect import *

# load model
model_root = os.getcwd() + '/models'
model_def = os.path.join(model_root, 'itracker_deploy.prototxt')
model_weights = os.path.join(model_root, 'snapshots/itracker25x_iter_92000.caffemodel')

net = caffe.Net(
    model_def,  # defines the structure of the model
    model_weights,  # contains the trained weights
    caffe.TEST)  # use test mode (e.g., don't perform dropout)


# set the batch size to 1
def set_batch_size(batch_size):
    net.blobs['image_left'].reshape(batch_size, 3, 224, 224)
    net.blobs['image_right'].reshape(batch_size, 3, 224, 224)
    net.blobs['image_face'].reshape(batch_size, 3, 224, 224)
    net.blobs['facegrid'].reshape(batch_size, 625, 1, 1)

set_batch_size(1)

# load the mean images
def get_mean_image(file_name):
    path = os.path.join(model_root, 'mean_images', file_name)
    image_mean = np.array(scipy.io.loadmat(path)['image_mean'])
    image_mean = image_mean.reshape(3, 224, 224)

    return image_mean.mean(1).mean(1)


mu_face = get_mean_image('mean_face_224.mat')
mu_left_eye = get_mean_image('mean_left_224.mat')
mu_right_eye = get_mean_image('mean_left_224.mat')


def create_image_transformer(layer_name, mean_image=None):
    transformer = caffe.io.Transformer(
        {layer_name: net.blobs[layer_name].data.shape})
    # move image channels to outermost dimension
    transformer.set_transpose(layer_name, (2, 0, 1))
    # subtract the dataset-mean value in each channel
    if mean_image is not None:
        transformer.set_mean(layer_name, mean_image)
    return transformer


left_eye_transformer = create_image_transformer('image_left', mu_left_eye)
right_eye_transformer = create_image_transformer('image_right', mu_right_eye)
face_transformer = create_image_transformer('image_face', mu_face)



def extract_gaze(face, face_feature):
    face_image, eye_images, face_grid = face_feature

    if len(eye_images) < 2:
        return None

    start_ms = current_time()
    transformed_right_eye = right_eye_transformer.preprocess('image_right', eye_images[0])
    transformed_left_eye = left_eye_transformer.preprocess('image_left', eye_images[1])
    transformed_face = face_transformer.preprocess('image_face', face_image)
    transformed_face_grid = np.copy(face_grid).reshape(1, 625, 1, 1)

    net.blobs['image_left'].data[...] = transformed_left_eye
    net.blobs['image_right'].data[...] = transformed_right_eye
    net.blobs['image_face'].data[...] = transformed_face
    net.blobs['facegrid'].data[...] = transformed_face_grid

    output = net.forward()
    net.forward()
    print("Detect gazes took " + str((current_time() - start_ms) * 1. / 1000) + "s")
    
    return np.copy(output['fc3'][0])

def extract_gazes(faces, face_features):
    outputs = []
    for i, face in enumerate(faces): 
        output = extract_gaze(face, face_features[i])
        if output is not None:
            outputs.append(output)
    return outputs



if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(0)

    # capture frame from camera
    cap = cv2.VideoCapture(0)

    img = None

    while (True):
        # get frame from camera
        ret, frame = cap.read()
        if ret:
            img = frame

        if img is not None:
            # detect face
            img, faces, face_features = extract_frame_features(img)

            # draw features
            render_face_feats(img, faces, face_features)

            # detect gaze
            outputs = extract_gazes(faces, face_features)

            print('gazes:', outputs)

            # draw features
            img = render_gazes(img, outputs)

            # display image
            cv2.imshow('frame', img)

        # ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('\x1b'):
            break

    cap.release()
    cv2.destroyAllWindows()
