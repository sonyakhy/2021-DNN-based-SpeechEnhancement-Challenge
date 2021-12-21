import os
import numpy as np
import librosa
import torch
import scipy.io.wavfile as wav
from tqdm import tqdm
from pathlib import Path
import soundfile
import matplotlib
# https://github.com/smothly/High_Perfomance_Python/blob/master/Untitled.ipynb

# 디렉토리를 읽고 그 디랙토리에 있는 파일들을 리스트에 담
def scan_directory(dir_name):
    """Scan directory and save address of clean/noisy wav data.
    Args:
        dir_name: directroy name to scan
    Returns:
        addr: all address list of clean/noisy wave data in subdirectory
    """
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()
    else:
        print("Scanning a directory %s " % dir_name)

    # addr = ["/Dataset/clean", "/Dataset/noise"]
    addr = []

    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"): # endswith는 file의 마지막 단어와 매치되는지 체크하고 True/False를 반환함.
                filepath = Path(subdir) / file
                addr.append(filepath)
    return addr

# 음성의 순서를 회전함.
def shifting_sound(data, sr=22050, roll_rate=0.1):
    # 그냥 [1, 2, 3, 4] 를 [4, 1, 2, 3]으로 만들어주는겁니다.
    data_roll = np.roll(data, int(len(data) * roll_rate))
    return data_roll

# Minus the sound : 위상을 뒤집는 것으로, 원래 소리와 똑같이 들림.
def minus_sound(data, sr=22050):
    # 위상을 뒤집는 것으로서 원래 소리와 똑같이 들린다.
    temp_numpy = (-1) * data

    return temp_numpy

# Reverse the sound
def reverse_sound(data, sr=22050):
    # 거꾸로 재생
    data = np.array([data[len(data) - 1 - i] for i in range(len(data))])
    return data

# main
clean_dir = Path("./Dataset/train/clean/")
addr_augmentation = Path("./Dataset/train/clean_aug/")

if os.path.isdir(clean_dir) is False:
    os.system('mkdir ' + str(clean_dir))

if os.path.isdir(addr_augmentation) is False:
    os.system('mkdir ' + str(addr_augmentation))

list_clean_files = scan_directory(clean_dir)

for addr_speech in list_clean_files:  # clean 디렉토리의 파일들을 불러옴
    wav_speech, read_fs = soundfile.read(addr_speech)  # data , sample rate

    # shift_data = shifting_sound(wav_speech, read_fs) # data shifting
    minus_data = minus_sound(wav_speech, read_fs)  # Minus the sound
    # Reverse_data = reverse_sound(wav_speech, read_fs)  # Reverse the sound
    augmentation_name = Path(addr_speech).name[:-4] + '_aug.wav' # augmentation data name : clean 데이터 마지막에 _aug를 붙임
    addr_augmentation_save = addr_augmentation / augmentation_name
    wav.write(addr_augmentation_save, read_fs, minus_data) # create wave file
