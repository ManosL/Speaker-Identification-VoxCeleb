import argparse
import os, sys
sys.path.append('./Utils')

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from read_data   import read_dataset_shortTermFeatures, read_dataset_wav2vec
from train_utils import cross_validate_model, train_and_evaluate_model_on_test_set
from preprocessing_utils import normalize_dataset

DATASET_PATH = '../../voxCelebDataset/vox1_dev_txt'

STD_THRESHOLD = 0.0002



def knn_preprocessing(train_instances, train_labels, test_instances, test_labels):
    scaler = MinMaxScaler()
    scaler.fit(train_instances)

    new_train_instances = scaler.transform(train_instances)
    new_test_instances  = scaler.transform(test_instances)

    return new_train_instances, train_labels, new_test_instances, test_labels



def check_dir_exists(dir_path):
    dir_path_str = str(dir_path)

    if not os.path.isdir(dir_path_str):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid directory" % dir_path_str)

    return dir_path_str



def main():
    global DATASET_PATH

    parser = argparse.ArgumentParser('description=Experiments Execution')

    parser.add_argument('--dataset_path', type=check_dir_exists, default=DATASET_PATH,
                        help='path to the VoxCeleb dataset where we can find the features')

    args = parser.parse_args()

    DATASET_PATH = args.dataset_path

    st_instances,  st_labels  = read_dataset_shortTermFeatures(DATASET_PATH)
    w2v_instances, w2v_labels = read_dataset_wav2vec(DATASET_PATH)

    # Train-Test Split
    st_train_instances, st_test_instances, \
    st_train_labels, st_test_labels         = train_test_split(st_instances, st_labels, 
                                                            stratify=st_labels)
    
    w2v_train_instances, w2v_test_instances, \
    w2v_train_labels, w2v_test_labels       = train_test_split(w2v_instances, w2v_labels, 
                                                            stratify=w2v_labels)
                                                
    description = st_train_instances.describe()

    # Remove constant features for shortTerm ones
    feature_names = description.columns
    feature_stds  = description.loc['std']
    
    print('Standard Deviations')
    for n, s in zip(feature_names, feature_stds):
        if s < STD_THRESHOLD:
            st_train_instances = st_train_instances.drop(n, axis=1)
            st_test_instances = st_test_instances.drop(n, axis=1)

    models_to_test   = []
    models_prep      = []
    model_names      = []

    n_neighbors_vals = [1, 5, 10, 25, 50, 100, 200, 350]

    train_accs = []
    val_accs   = []
    test_accs  = []

    for n_neighbors in n_neighbors_vals:
        models_to_test.append(KNeighborsClassifier(n_neighbors, metric='cosine'))
        models_prep.append(knn_preprocessing)
        model_names.append('kNN with k=' + str(n_neighbors))

    for model, model_prep_fn in zip(models_to_test, models_prep):
        st_train_acc, st_val_acc      = cross_validate_model(model, st_train_instances, \
                                                            st_train_labels, model_prep_fn)
        
        _, st_test_acc   = train_and_evaluate_model_on_test_set(model, \
                                                st_train_instances, st_train_labels, \
                                                st_test_instances, st_test_labels, \
                                                model_prep_fn)

        w2v_train_acc, w2v_val_acc = cross_validate_model(model, w2v_train_instances, \
                                                        w2v_train_labels, model_prep_fn)
        
        _, w2v_test_acc            = train_and_evaluate_model_on_test_set(model, \
                                                        w2v_train_instances, w2v_train_labels, \
                                                        w2v_test_instances, w2v_test_labels, \
                                                        model_prep_fn)

        train_accs.append([st_train_acc, w2v_train_acc])
        val_accs.append([st_val_acc, w2v_val_acc])
        test_accs.append([st_test_acc, w2v_test_acc])


    feature_types = ['shortTermFeatures', 'Wav2Vec2']
    print('Train Accuracies')
    print(pd.DataFrame(train_accs, index=model_names, columns=feature_types))

    print()
    print()
    print('Validation Accuracies')
    print(pd.DataFrame(val_accs, index=model_names, columns=feature_types))

    print()
    print()
    print('Test Accuracies')
    print(pd.DataFrame(test_accs, index=model_names, columns=feature_types))

    return 0



if __name__ == '__main__':
    main()