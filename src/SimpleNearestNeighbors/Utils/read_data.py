import os
import pickle

import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve

import torch
import torchaudio



# For each instance we will get the average of each window of those
# features
def read_dataset_shortTermFeatures(dataset_path):
    print(dataset_path)
    assert(os.path.isdir(dataset_path))

    column_names = None
    instances    = []
    labels       = []

    features_dir = os.path.join(dataset_path, 'features')

    celebs = os.listdir(features_dir)
    celebs_num = len(celebs)

    for i, celeb in zip(range(celebs_num), celebs):
        curr_label = celeb

        celeb_dir = os.path.join(features_dir, celeb)
        video_ids = os.listdir(celeb_dir)

        for video_id in video_ids:
            celeb_video_dir = os.path.join(celeb_dir, video_id)

            samples = os.listdir(celeb_video_dir)

            for sample in samples:
                sample_dir         = os.path.join(celeb_video_dir, sample)
                features_file_path = os.path.join(sample_dir, 'shortTermFeatures.pickle')

                if not os.path.exists(features_file_path):
                    continue
                
                file_size = os.path.getsize(features_file_path)

                if file_size == 0:
                    continue

                features_file      = open(features_file_path, 'rb')
                [st_features, st_feature_names] = pickle.load(features_file)
                
                st_features = st_features.T
                
                if column_names != None:
                    assert(column_names == st_feature_names)
                else:
                    column_names = st_feature_names
                
                st_features = list(np.average(st_features, axis=0))
                instances.append(st_features)
                labels.append(curr_label)

                features_file.close()
    
        if (i + 1) % 20 == 0:
            print('Loaded ' + str(i + 1) + '/' + str(celebs_num) + ' Celebrities.')

    print("Successfully Loaded the Dataset")

    instances = pd.DataFrame(instances, columns=column_names)
    labels    = pd.Series(labels)

    return instances, labels



# For each instance we will get the average of each window of those
# features
def read_dataset_wav2vec(dataset_path):
    assert(os.path.isdir(dataset_path))

    column_names = None
    instances    = []
    labels       = []

    features_dir = os.path.join(dataset_path, 'features')

    celebs = os.listdir(features_dir)
    celebs_num = len(celebs)

    for i, celeb in zip(range(celebs_num), celebs):
        curr_label = celeb

        celeb_dir = os.path.join(features_dir, celeb)
        video_ids = os.listdir(celeb_dir)

        for video_id in video_ids:
            celeb_video_dir = os.path.join(celeb_dir, video_id)

            samples = os.listdir(celeb_video_dir)

            for sample in samples:
                sample_dir         = os.path.join(celeb_video_dir, sample)
                features_file_path = os.path.join(sample_dir, 'wav2vecFeatures.pickle')

                if not os.path.exists(features_file_path):
                    continue

                file_size = os.path.getsize(features_file_path)

                if file_size == 0:
                    continue

                features_file    = open(features_file_path, 'rb')
                wav2vec_features = pickle.load(features_file)
                wav2vec_features = wav2vec_features[0]          # I NEED IT JUST FOR NOW
                                                                # IF I RUN AGAIN THE READ I DO NOT NEED IT

                wav2vec_features = list(np.average(wav2vec_features, axis=0))

                instances.append(wav2vec_features)
                labels.append(curr_label)

                features_file.close()
    
        if (i + 1) % 20 == 0:
            print('Loaded ' + str(i + 1) + '/' + str(celebs_num) + ' Celebrities.')

    print("Successfully Loaded the Dataset")

    instances = pd.DataFrame(instances, columns=column_names)
    labels    = pd.Series(labels)

    return instances, labels



def get_id_to_celeb_name_map(metadata_path):
    metadata_file = open(metadata_path, 'r')

    lines   = metadata_file.readlines()
    mapping = {}

    for line in lines[1:]:
        id   = line.split()[0]
        name = line.split()[1]

        assert(id not in mapping.keys())

        mapping[id] = name
    
    return mapping
