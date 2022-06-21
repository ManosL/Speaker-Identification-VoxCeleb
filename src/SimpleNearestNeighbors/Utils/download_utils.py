import os
import pickle

from math import ceil, floor
import scipy.io.wavfile as wavfile
import numpy as np

from pytube import YouTube
import moviepy.editor as mp

import librosa
from pyAudioAnalysis import ShortTermFeatures, MidTermFeatures



BASE_LINK     = 'https://www.youtube.com/watch?v='
TMP_SAVE_PATH = './'

def __download_youtube_video(video_id):
    link = BASE_LINK + video_id

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

    # Filtering >= 25fps videos in that way because if I add an extra argument on
    # the above call it does not get recognized
    mp4_streams = list(filter(lambda s: s.fps >= 25, mp4_streams))

    if len(mp4_streams) == 0:
        print('Progressive stream with mp4 extension and frame rate more than 25fps was not found for ' + link)
        return -1
    
    # Get the first progressive stream of the filtered ones
    stream = mp4_streams[0]

    try:
        stream.download(output_path=TMP_SAVE_PATH, filename=video_id + '.mp4')
    except Exception as e:
        print(link + ' raised ' + str(e) +' when I tried to download it')
        return -1 

    return 0



def download_video_and_audio_extraction(video_id, start_second, end_second):
    assert(start_second < end_second)
    tmp_mp4_path = os.path.join(TMP_SAVE_PATH, video_id + '.mp4')

    if not os.path.exists(tmp_mp4_path):
        status = __download_youtube_video(video_id)

        if status != 0:
            print('Could not download video at link ' + BASE_LINK + video_id)
            return -1, None

    my_clip = mp.VideoFileClip(tmp_mp4_path)
    my_clip = my_clip.set_fps(25)
    # You need to take a subclip via .subclip command

    if start_second > my_clip.duration:
        print(video_id, 'has start_sec > duration for start second equal to', start_second)

        return -2, None

    my_clip = my_clip.subclip(start_second, end_second)

    return 0, my_clip.audio