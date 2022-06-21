"""
Demo will work on the best model I found through experiments
"""
import argparse
import os, sys
sys.path.append('./Utils')

import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from read_data   import read_dataset_shortTermFeatures, get_id_to_celeb_name_map
from train_utils import cross_validate_model, train_and_evaluate_model_on_test_set
from preprocessing_utils import normalize_dataset

from pytube import YouTube
import moviepy.editor as mp

import librosa
from pyAudioAnalysis import ShortTermFeatures

DATASET_PATH       = '../voxCelebDataset/vox1_dev_txt'
METADATA_FILE_PATH = '../voxCelebDataset/vox1_meta.csv'

STD_THRESHOLD      = 0.0002



def download_video(link):
    # object creation using YouTube
    # which was imported in the beginning 
    print('\tDownloading ' + link)

    try:
        yt = YouTube(link)
    except Exception as e:
        print(link + ' raised ' + str(e) +' when I tried to download it')
        return -1 

    # I want to get the progressive streams in order to capture both the video and the 
    # audio in a single stream
    try:
        mp4_streams = yt.streams.filter(progressive=True, file_extension='mp4')
    except Exception as e:
        print(link + ' raised ' + str(e) +' when I tried to download it')
        return -1 
    
    # Get the first progressive stream of the filtered ones
    stream = mp4_streams[0]

    try:
        stream.download(output_path='./', filename='tmp.mp4')
    except Exception as e:
        print(link + ' raised ' + str(e) +' when I tried to download it')
        return -1 

    print('\tDownload finished for ' + link)
    return 0



def audio_extraction(youtube_link, start_second, duration):
    status = download_video(youtube_link)

    if status != 0:
        print('\tCould not download video at link ' + youtube_link)
        return -1, None

    tmp_mp4_path = './tmp.mp4'

    my_clip = mp.VideoFileClip(tmp_mp4_path)
    my_clip = my_clip.set_fps(25)

    # You need to take a subclip via .subclip command
    if start_second > my_clip.duration:
        print('\t', youtube_link, 'has start_sec > duration.')
        print('\tThis can happen because the original video was trimmed now or you gave wrong value.')

        return -2, None

    end_second = start_second + duration

    if end_second > my_clip.duration:
        end_second = my_clip.duration

    my_clip = my_clip.subclip(start_second, end_second)

    return 0, my_clip.audio



# We do not need to use pickle as the model is a quick one to train
# and also there will be the case that we might have more data
def train_model():
    # Read the instances
    st_instances,  st_labels  = read_dataset_shortTermFeatures(DATASET_PATH)

    # Remove constant features
    columns_to_drop = []
    description     = st_instances.describe()

    feature_names = description.columns
    feature_stds  = description.loc['std']
    
    for n, s in zip(feature_names, feature_stds):
        if s < STD_THRESHOLD:
            st_instances = st_instances.drop(n, axis=1)

            columns_to_drop.append(n)
    
    # Normalize the dataset using MinMax Scaler
    scaler = MinMaxScaler()
    scaler.fit(st_instances)

    st_instances = scaler.transform(st_instances)

    # Define and train our model
    n_neighbors = 1
    knn_model   = KNeighborsClassifier(n_neighbors, metric='cosine')

    knn_model.fit(st_instances, st_labels)

    return knn_model, columns_to_drop, scaler



def feature_extraction(audio_clip, columns_to_drop, scaler):
    Fs = 8000

    audio_clip.write_audiofile('./tmp.wav', verbose=False, logger=None)
    signal, _ = librosa.load('./tmp.wav', sr=Fs)

    os.remove('./tmp.wav')
    os.remove('./tmp.mp4')

    [st_features, st_feature_names] = ShortTermFeatures.feature_extraction(signal, Fs, int(Fs * 0.050), int(Fs * 0.050))
    
    st_features = st_features.T
    st_features = pd.DataFrame(st_features, columns=st_feature_names)
    st_features = st_features.drop(columns_to_drop, axis=1)

    st_features = st_features.to_numpy()

    st_features = np.average(st_features, axis=0)

    return scaler.transform([st_features])[0]



def predict_celebrity(model, feature_vector, id_to_name_map):
    pred_id = model.predict([feature_vector])[0]

    return id_to_name_map[pred_id]



def check_dir_exists(dir_path):
    dir_path_str = str(dir_path)

    if not os.path.isdir(dir_path_str):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid directory" % dir_path_str)

    return dir_path_str


def check_file_exists(file_path):
    file_path_str = str(file_path)

    if not os.path.isfile(file_path_str):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid file" % file_path_str)

    return file_path_str



def main():
    global DATASET_PATH
    global METADATA_FILE_PATH

    parser = argparse.ArgumentParser('description=Experiments Execution')

    parser.add_argument('--dataset_path', type=check_dir_exists, default=DATASET_PATH,
                        help='path to the VoxCeleb dataset where we can find the features')
    parser.add_argument('--metadata_path', type=check_file_exists, default=METADATA_FILE_PATH,
                        help='path to the VoxCeleb meta data file path')

    args = parser.parse_args()

    DATASET_PATH       = args.dataset_path
    METADATA_FILE_PATH = args.metadata_path

    id_to_names_map = get_id_to_celeb_name_map(METADATA_FILE_PATH)
    model, columns_to_drop, scaler = train_model()

    video_link = None

    while True:
        video_link = input('Give the link of the youtube video you want me to predict the speaker(write q to quit): ')

        if video_link == 'q':
            break
        
        print()
        start_second = int(input('\tGive the start second that our model will consider (should be less than the duration of video): '))
        
        while start_second < 0:
            print()
            print('Start second should be positive')
            start_second = int(input('\tGive the start second that our model will consider \
                                    (should be less than the duration of video): '))

        print()
        duration = int(input('\tGive the duration to consider after start second: '))

        while duration < 0:
            print()
            print('Duration should be positive')
            duration = int(input('\tGive the duration to consider after start second: '))

        _, audio_clip = audio_extraction(video_link, start_second, duration) 

        feature_vector = feature_extraction(audio_clip, columns_to_drop, scaler)
        celeb_name     = predict_celebrity(model, feature_vector, id_to_names_map)

        print('I believe that the speaker is', ' '.join(celeb_name.split('_')))

    return 0



if __name__ == '__main__':
    main()