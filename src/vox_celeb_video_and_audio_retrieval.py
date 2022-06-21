# TODO: get the audio of a subclip that is specified in the metadata file
import argparse
import os, sys
import shutil
import threading
import pickle

from math import ceil, floor
import scipy.io.wavfile as wavfile
import numpy as np

from pytube import YouTube
import moviepy.editor as mp

import librosa
from pyAudioAnalysis import ShortTermFeatures, MidTermFeatures

from PIL import Image

import torch, torchaudio

METADATA_PATH       = './voxCelebDataset/vox1_dev_txt'
METADATA_FPS        = 25

TMP_SAVE_PATH       = './tmp'
BASE_LINK           = 'https://www.youtube.com/watch?v='

DEVICE              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL               = None
MODEL_SAMPLE_RATE   = None

# Multithreading variables
print_sem     = threading.Semaphore()  # Semaphore for stdout
access_sem    = threading.Semaphore()
non_empty_sem = threading.Semaphore(0)

shared_queue = []



def sync_print(*args):
    print_sem.acquire()

    print(*args, flush=True)

    print_sem.release()

    return



def initialize_wav2vec_model():
    bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
    model  = bundle.get_model().to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    return model, bundle.sample_rate



def download_video(video_id):
    link = BASE_LINK + video_id

    # object creation using YouTube
    # which was imported in the beginning 
    sync_print('Downloading ' + link)

    try:
        yt = YouTube(link)
    except Exception as e:
        sync_print(link + ' raised ' + str(e) +' when I tried to download it')
        return -1 

    # I want to get the progressive streams in order to capture both the video and the 
    # audio in a single stream
    try:
        mp4_streams = yt.streams.filter(progressive=True, file_extension='mp4')
    except Exception as e:
        sync_print(link + ' raised ' + str(e) +' when I tried to download it')
        return -1 

    # Filtering >= 25fps videos in that way because if I add an extra argument on
    # the above call it does not get recognized
    mp4_streams = list(filter(lambda s: s.fps >= 25, mp4_streams))

    if len(mp4_streams) == 0:
        sync_print('Progressive stream with mp4 extension and frame rate more than 25fps was not found for ' + link)
        return -1
    
    # Get the first progressive stream of the filtered ones
    stream = mp4_streams[0]

    try:
        stream.download(output_path=TMP_SAVE_PATH, filename=video_id + '.mp4')
    except Exception as e:
        sync_print(link + ' raised ' + str(e) +' when I tried to download it')
        return -1 

    sync_print('Download finished for ' + link)
    return 0



def download_video_and_audio_extraction(video_id, start_frame=None, end_frame=None):
    tmp_mp4_path = os.path.join(TMP_SAVE_PATH, video_id + '.mp4')

    if not os.path.exists(tmp_mp4_path):
        status = download_video(video_id)

        if status != 0:
            sync_print('Could not download video at link ' + BASE_LINK + video_id)
            return -1, None

    my_clip = mp.VideoFileClip(tmp_mp4_path)
    my_clip = my_clip.set_fps(25)
    # You need to take a subclip via .subclip command

    start_second = 0

    if start_frame != None:
        start_second = floor(start_frame / METADATA_FPS)

    end_second = my_clip.duration

    if end_frame != None:
        end_second = ceil(end_frame / METADATA_FPS)

    if start_second > my_clip.duration:
        print(video_id, 'has start_sec > duration for start and end frame', start_frame, end_frame)
        print('This can happen because the original video was trimmed now.')

        return -2, None

    my_clip = my_clip.subclip(start_second, end_second)

    return 0, my_clip.audio



def delete_video_from_tmp_dir(video_id):
    file_to_delete = os.path.join(TMP_SAVE_PATH, video_id + '.mp4')

    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)

    return



def audio_already_extracted(celeb_id, video_id):
    txt_files_path = os.path.join(METADATA_PATH, 'txt', celeb_id, video_id)
    wav_files_path = os.path.join(METADATA_PATH, 'wav', celeb_id, video_id)

    if not os.path.isdir(wav_files_path):
        return False

    return True if len(os.listdir(txt_files_path)) == len(os.listdir(wav_files_path)) else False



def txt_get_start_and_end_frame(path):
    lines = open(path, 'r').readlines()

    frames_table_line_index = None
    
    for i, line in zip(range(len(lines)), lines):
        if line.startswith('FRAME'):
            frames_table_line_index = i
            break
    
    assert(frames_table_line_index != None)

    frame_lines = lines[frames_table_line_index + 1:]

    start_frame = int(frame_lines[0].split()[0])
    end_frame   = int(frame_lines[-1].split()[0])

    sync_print(path, start_frame, end_frame)
    assert(len(frame_lines) == (end_frame - start_frame + 1))

    return start_frame, end_frame



def save_wav(path_to_save, audio_clip):
    audio_clip.audio.write_audiofile(path_to_save, verbose=False, logger=None)

    return



def normalize_signal(signal):
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    signal = (signal - signal.mean())
    return  signal / ((np.abs(signal)).max() + 0.0000000001)



# I will store the spectograms and the features that pyAudioAnalysis
# or librosa extract
def extract_and_save_features(dir_to_save, wav_file_path, sampling_freq=8000):
    # use librosa's load to get the wav to the desired sampling frequency
    signal, Fs = librosa.load(wav_file_path, sr=sampling_freq)

    assert(Fs == sampling_freq)

    signal = normalize_signal(signal)

    # Save normalized signal in order to handle it as you want later
    signal_path = os.path.join(dir_to_save, 'signal.pickle')

    if not os.path.isfile(signal_path):
        signal_file = open(signal_path, 'wb')
        pickle.dump(tuple([signal, Fs]), signal_file)
        signal_file.close()

    # Get short term features using the pyAudioAnalysis as they should be saved
    # I extract them because I'll surely use them
    st_features_path = os.path.join(dir_to_save, 'shortTermFeatures.pickle')

    if not os.path.isfile(st_features_path):
        st_features_file = open(st_features_path, 'wb')
        [st_features, st_feature_names] = ShortTermFeatures.feature_extraction(signal, Fs, int(Fs * 0.050), int(Fs * 0.050))
        
        pickle.dump(tuple([st_features, st_feature_names]), st_features_file)
        st_features_file.close()
    
    # Getting wav2vec features
    w2v_features_path = os.path.join(dir_to_save, 'wav2vecFeatures.pickle')

    if not os.path.isfile(w2v_features_path):
        w2v_features_file = open(w2v_features_path, 'wb')

        if sampling_freq != MODEL_SAMPLE_RATE:
            signal = torchaudio.functional.resample(torch.tensor(signal), sampling_freq, MODEL_SAMPLE_RATE)

        input_tensor = torch.tensor(signal)
        input_tensor = torch.reshape(input_tensor, (1, -1))
        input_tensor = input_tensor.type(torch.FloatTensor).to(DEVICE)

        wav2vec_features, _ = MODEL(input_tensor)
        wav2vec_features    = wav2vec_features.cpu().detach().numpy()
        
        pickle.dump(wav2vec_features[0], w2v_features_file)
        
        del input_tensor
        torch.cuda.empty_cache()
        
        w2v_features_file.close()
    
    return



"""
 Readers-Writer solution where only one reader or one writer can access at the time

 R

    down(non_empty_sem)
    down(access_sem)

    // READ FROM QUEUE

    up(access_sem)

 W

    down(access_sem)

    //WRITE TO QUEUE


    up(non_empty_sem)
    up(access_sem)    
"""
# Worker thread that will read the celebrity directory and will download all its videos
# and get its audio samples
def worker(id, txt_data_path, wav_to_create_path, features_dir):
    sync_print('Started worker', id)

    while 1:
        non_empty_sem.acquire()
        access_sem.acquire()

        celeb_dir = shared_queue.pop(0)

        access_sem.release()

        if celeb_dir == 'END':
            break
        
        wav_celeb_dir = os.path.join(wav_to_create_path, celeb_dir)

        if not os.path.isdir(wav_celeb_dir):
            os.mkdir(wav_celeb_dir)

        features_celeb_dir = os.path.join(features_dir, celeb_dir)

        if not os.path.isdir(features_celeb_dir):
            os.mkdir(features_celeb_dir)

        celeb_dir_path = os.path.join(txt_data_path, celeb_dir)
        celeb_videos   = sorted(os.listdir(celeb_dir_path))

        for celeb_video_ref in celeb_videos:
            need_to_download = True

            if audio_already_extracted(celeb_dir, celeb_video_ref):
                # We already extracted the audio of all samples for that
                # video, thus, we do not need to download it again and
                # extract the audio samples that we want

                # But we need to get through the wav files and extract the
                # features 
                need_to_download = False
            
            wav_celeb_video_dir = os.path.join(wav_celeb_dir, celeb_video_ref)
            
            if not os.path.isdir(wav_celeb_video_dir):
                os.mkdir(wav_celeb_video_dir)

            features_celeb_video_dir = os.path.join(features_celeb_dir, celeb_video_ref)
            
            if not os.path.isdir(features_celeb_video_dir):
                os.mkdir(features_celeb_video_dir)

            video_samples_dir_path = os.path.join(celeb_dir_path, celeb_video_ref)
            video_samples          = sorted(os.listdir(video_samples_dir_path))

            for sample in video_samples:
                curr_file = os.path.join(video_samples_dir_path, sample)

                path_to_save = os.path.join(wav_celeb_video_dir, sample.split('.')[0] + '.wav')

                if need_to_download == True:
                    start_frame, end_frame = txt_get_start_and_end_frame(curr_file)

                    sync_print(celeb_dir, celeb_video_ref, sample, start_frame, end_frame)
                    status, wav_clip = download_video_and_audio_extraction(celeb_video_ref, start_frame, end_frame)

                    if status == -1:
                        break

                    save_wav(path_to_save, wav_clip)

                sample_features_dir = os.path.join(features_celeb_video_dir, sample.split('.')[0])

                if not os.path.isdir(sample_features_dir):
                    os.mkdir(sample_features_dir)

                extract_and_save_features(sample_features_dir, path_to_save)

            delete_video_from_tmp_dir(celeb_video_ref)

    sync_print('Worker', id, 'exits')

    return 0




def check_positive_int(value):
    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    
    return ivalue



def check_dir_exists(dir_path):
    dir_path_str = str(dir_path)

    if not os.path.isdir(dir_path_str):
        raise argparse.ArgumentTypeError("\'%s\' is not a valid directory" % dir_path_str)

    return dir_path_str



def main():
    global METADATA_PATH

    parser = argparse.ArgumentParser('description=VoxCeleb Audio Reader + Feature Extractor')

    parser.add_argument('--dataset_path', type=check_dir_exists, default=METADATA_PATH,
                        help='path to the VoxCeleb dataset(it should have the form that voxCeleb has.')
    parser.add_argument('--workers_num', type=check_positive_int, default=4,
                        help='number of thread that will concurrently read the dataset')
    parser.add_argument('--celebs_num', type=check_positive_int, default=50,
                        help='Number of celebrities to read(it will read the first N celebrities)')
    parser.add_argument('--log', type=str, default=None,
                        help='Specify a log file to redirect stdout')

    args = parser.parse_args()

    METADATA_PATH = args.dataset_path

    if os.path.isdir(TMP_SAVE_PATH):
        shutil.rmtree(TMP_SAVE_PATH)

    os.mkdir(TMP_SAVE_PATH)

    data_root_dir = os.path.join(METADATA_PATH, 'txt')
    wav_data_dir  = os.path.join(METADATA_PATH, 'wav')
    features_dir  = os.path.join(METADATA_PATH, 'features')

    if not os.path.isdir(wav_data_dir):
        os.mkdir(wav_data_dir)

    if not os.path.isdir(features_dir):
        os.mkdir(features_dir)

    global MODEL
    global MODEL_SAMPLE_RATE

    MODEL, MODEL_SAMPLE_RATE = initialize_wav2vec_model()

    threads = []

    for i in range(args.workers_num):
        thread_args = (i, data_root_dir, wav_data_dir, features_dir)

        threads.append(threading.Thread(target=worker, args=thread_args))
        threads[-1].start()

    celebrities_dirs  = sorted(os.listdir(data_root_dir))[0:args.celebs_num]

    for celeb_dir in celebrities_dirs:
        access_sem.acquire()

        shared_queue.append(celeb_dir)

        non_empty_sem.release()
        access_sem.release()

    for i in range(args.workers_num):
        access_sem.acquire()

        shared_queue.append('END')

        non_empty_sem.release()
        access_sem.release()
    
    for i in range(args.workers_num):
        threads[i].join()

    shutil.rmtree(TMP_SAVE_PATH)



if __name__ == "__main__":
    main()