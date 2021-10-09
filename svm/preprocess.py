# import utility functions
from utils_frame_based import *
import pandas as pd
import numpy as np
import os

# define three constants.
# You can later experiment with these constants by changing them to adaptive variables.
EAR_THRESHOLD = 0.21  # eye aspect ratio to indicate blink
EAR_CONSEC_FRAMES = 3  # number of consecutive frames the eye must be below the threshold
SKIP_FIRST_FRAMES = 150  # how many frames we should skip at the beggining

# create a folder named 'train'
os.mkdir('./train')

# read all videos
directory = "./eyeblink8"
subjects = os.listdir(directory)
for subject in subjects:
    video_names = os.listdir(directory + '/' + subject)
    for video_name in video_names:
        clean_name = os.path.splitext(video_name)[0]
        extension = os.path.splitext(video_name)[1]
        if extension == '.avi':
            file_path = directory + '/' + subject + '/' + video_name
            print(file_path)
            frame_info_df, video_info_dict = process_video(
                file_path,
                subject=subject,
                external_factors=None,
                facial_actions=clean_name,
                ear_th=EAR_THRESHOLD,
                consec_th=EAR_CONSEC_FRAMES,
                skip_n=SKIP_FIRST_FRAMES)
            frame_info_df.to_pickle('./train/{}_{}_frame_info_df.pkl'.format(
                subject, clean_name))
            video_info_dict.to_pickle('./train/{}_{}_video_info_df.pkl'.format(
                subject, clean_name))

# read annotations
for subject in subjects:
    video_names = os.listdir(directory + '/' + subject)
    for video_name in video_names:
        clean_name = os.path.splitext(video_name)[0]
        extension = os.path.splitext(video_name)[1]
        if extension == '.tag':
            file_path = directory + '/' + subject + '/' + video_name
            print(file_path)
            #length of video
            frame_info_df = pd.read_pickle("./train/" + subject + '_' +
                                           clean_name + "_frame_info_df.pkl")
            len_video = len(frame_info_df)
            # read tag file
            annot_df = read_annotations(file_path, len_video)
            annot_df.to_pickle('./train/{}_{}_annotations.pkl'.format(
                subject, clean_name))

merge_pickles("./train")
concat_pickles("./train", "merged_df", "training_set")

print('# Train set is done.')
