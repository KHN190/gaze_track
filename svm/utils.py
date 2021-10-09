import os
import pandas as pd
import numpy as np

# define read_annotations
def read_annotations(input_file, len_video):
    # Read .tag file using readlines()
    with open(input_file) as f:
        lines = f.readlines()

    # find "#start" line
    start_line = 1
    for line in lines:
        clean_line = line.strip()
        if clean_line == "#start":
            break
        start_line += 1

    # length of annotations
    len_annot = len(lines[start_line:-1])  # -1 since last line will be"#end"

    blink_list = [0] * len_video
    closeness_list = [0] * len_video

    # convert tag file to readable format and build "closeness_list" and "blink_list"
    for i in range(len_annot):
        annotation = lines[start_line + i].split(':')

        if int(annotation[1]) > 0:
            # it means a new blink
            blink_frame = int(annotation[0])
            blink_list[blink_frame] = 1

        # if current annotation consist fully closed eyes, append it also to "closeness_list"
        if annotation[3] == "C" and annotation[5] == "C":
            closed_frame = int(annotation[0])
            closeness_list[closed_frame] = 1

    result_df = pd.DataFrame(list(zip(closeness_list, blink_list)),
                             columns=['closeness_annot', 'blink_annot'])
    return result_df


def merge_pickles(directory):
    annots = []
    frame_infos = []
    video_infos = []

    files = os.listdir(directory)
    for file in files:
        clean_name = os.path.splitext(file)[0]
        if clean_name.endswith('annotations'):
            annots.append(file)
        if clean_name.endswith('video_info_df'):
            video_infos.append(file)
        if clean_name.endswith('frame_info_df'):
            frame_infos.append(file)

    for file in annots:
        clean_name = os.path.splitext(file)[0]
        first_part = clean_name[:-12]

        for file2 in frame_infos:
            clean_name2 = os.path.splitext(file2)[0]
            first_part2 = clean_name2[:-14]
            if first_part == first_part2:
                frame_info_df = pd.read_pickle(directory + '/' + file2)
                annotation = pd.read_pickle(directory + '/' + file)
                if len(frame_info_df) != len(annotation):
                    os.mkdir(directory + '/fix/')
                    os.rename(directory + file, directory + '/fix/' + file)
                    os.rename(directory + file2, directory + '/fix/' + file2)
                    print(file2, len(frame_info_df))
                    print(file, len(annotation))
                else:
                    result = pd.concat([frame_info_df, annotation], axis=1)
                    result.to_pickle(directory + '/' + first_part +
                                     '_merged_df.pkl')


# append all of pickles ending particular string (i.e. "merged_df") in a directory
def concat_pickles(directory, ending, output_name):
    pickles = os.listdir(directory)
    pickle_list = []

    for pickle_file in pickles:
        clean_name = os.path.splitext(pickle_file)[0]
        if clean_name.endswith(ending):
            pickle = pd.read_pickle(directory + '/' + pickle_file)
            pickle_list.append(pickle)

    result = pd.concat(pickle_list)
    result.reset_index(inplace=True, drop=True)
    result.to_pickle(directory + '/' + output_name + '.pkl')


def build_svm_data(train_set, frames=13):
    # read dataset
    df = pd.read_pickle(train_set)
    ear = df[['subject', 'frame_no', 'avg_ear', 'blink_annot']]

    # group by subject
    user_list = tuple(ear['subject'].unique())
    list_of_dfs = []
    for user in user_list:
        list_of_dfs.append(ear.groupby('subject').get_group(user))

    # construct a df of continuous frame window
    win = frames // 2
    win_end = (win + 1) if frames % 2 == 1 else win

    dfs = []
    for subject_df in list_of_dfs:
        for i in range(-win, win_end):
            subject_df['ear' + str(i)] = subject_df.shift(\
                periods=i * (-1))['avg_ear']
        subject_df = subject_df[6:-6]
        dfs.append(subject_df)

    # concat results
    ear_df = pd.concat(dfs)
    ear_df.drop(columns=['frame_no', 'avg_ear'], inplace=True)
    ear_df.reset_index(drop=True, inplace=True)

    # train set for svm
    y = ear_df.loc[:, 'blink_annot'].values
    X = ear_df.drop(columns=['blink_annot'])

    return (X, y)


# df to normalized (0, 1) data
def transform_svm_data(X):
    from sklearn.preprocessing import StandardScaler

    f = lambda x: (StandardScaler().fit_transform(x.to_frame()))[:, 0]

    norm_X = X.groupby('subject').transform(f)
    norm_X = StandardScaler().fit_transform(norm_X)

    return norm_X


def save_model(svm, name):
    import pickle as pkl

    with open(name, 'wb+') as f:
        pkl.dump(svm, f)

    print(f"Model saved to {name}")


def load_model(name):
    import pickle as pkl

    with open(name, 'rb') as f:
        return pkl.load(f)
