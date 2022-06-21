# Speaker-Identification

## Overview

In this project I build a simple Speaker Identifier by using kNN. Also in the future I will keep there more approaches for this problem.

## Manual

### Installation instructions
In order to run the experiments in your local machine you should do the following steps.

1. Clone the repo by running `git clone https://github.com/ManosL/UFC-MMA-Fight-Predictor.git`
2. Afterwards, install virtualenv in pip3(if you did not do that already) by running
`pip3 install virtualenv`
3. Then move to this repository directory.
4. Then create and activate the virtual environment by running the following commands
```
virtualenv <venv_name>
source ./<venv_name>/bin/activate
```
5. Finally install the requirements by running `pip3 install -r requirements.txt`
6. You are ready to move to `src/` directory and run the experiments and demo programs!

### Dataset Download and Audio and Features retrieval instructions

In order to run the experiments or the demo you need to have the VoxCeleb Dataset along with the necessary pickle files. The instructions to do that are the following:

1. Download VoxCeleb dataset from [this page](https://mm.kaist.ac.kr/datasets/voxceleb/#downloads). Do NOT download the audio files.
2. Run the `src/vox_celeb_video_and_audio_retrieval.py` script that reads the dataset's metadata and creates a `wav/` and `feature/` directory with the same structure as the `txt/` directory that contain the audio and features of each sample, respectively. The script should be run with the following way:
```
        python3 vox_celeb_video_and_audio_retrieval.py --dataset_path <VoxCelebDataset_path> --workers_num <workers_num> --celebs_num <celebs_num> (--log <log_file>)
```
where:
- <VoxCelebDataset_path> is the path to the VoxCeleb Dataset.
- <workers_num> is the number of threads that will work concurrently to retrieve the dataset.
- <celebs_num> is the number of celebrities to read.(Reads the first <celebs_num> celebrities)
- <log_file> is where the program will write any logs. If not provided, it is written in stdout.

NOTE: This script works by downloading the youtube videos initially and then extracts the audio sample and from that extracts the features.

### Experiments instructions

In order to run the experiments done in order to write the report, go into `src/SimpleNearestNeighbors/` directory and run the following
command:

```
        python3 experiments.py --dataset_path <VoxCelebDataset_path>
```

where:
- <VoxCelebDataset_path> is the path to the VoxCeleb Dataset.

WARNING: This will take time in order to complete.

### Demo instructions

In order to run the demo you should move again into the `src/SimpleNearestNeighbors/` directory and run the following command:
```
python3 demo.py --dataset_path <VoxCelebDataset_path> --metadata_path <VoxCelebMetaData_path>
```
where:
- <VoxCelebDataset_path> is the path to the VoxCeleb Dataset in order to read the instances and then train our model.
- <VoxCelebMetaData_path> is the path to VoxCeleb's dataset path to metadat file where we can extract a mapping from celebrity's ID to its name.