"""
generate noisy data with various noise files
"""
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import librosa
from pathlib import Path
import soundfile

#######################################################################
#                         data info setting                           #
#######################################################################
# USE THIS, OR SYS.ARGVS
# mode = 'validation'  # train / validation / test
# snr_set = [0, 5]
# fs = 16000
#######################################################################
#                                main                                 #
#######################################################################
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


# Generate noisy data given speech, noise, and target SNR.
def generate_noisy_wav(wav_speech, wav_noise, snr): # clean / noise / snr => clean과 noise를 결합하여 snr의 noisy 데이터를 만들어냄.
    # Obtain the length of speech and noise components.
    len_speech = len(wav_speech) # clean 파일의 길이 48000
    len_noise = len(wav_noise) # noise 파일의 길이 160000
    # clean과 noise의 길이를 같게 해줘야함

    # Select noise segment randomly to have same length with speech signal.
    # clean 파일과 똑같은 길이를 가지도록 랜덤으로 noise 파일의 구간을 지정함.
    st = np.random.randint(0, len_noise - len_speech) # st : 73637
    ed = st + len_speech # ed : 121637
    wav_noise = wav_noise[st:ed] # 결론적으로 noise 파일의 길이는 clean과 동일해짐

    # Compute the power of speech and noise after removing DC bias.
    dc_speech = np.mean(wav_speech)
    dc_noise = np.mean(wav_noise)
    pow_speech = np.mean(np.power(wav_speech - dc_speech, 2.0))
    pow_noise = np.mean(np.power(wav_noise - dc_noise, 2.0))

    # Compute the scale factor of noise component depending on the target SNR.
    alpha = np.sqrt(10.0 ** (float(-snr) / 10.0) * pow_speech / (pow_noise + 1e-6))
    noisy_wav = (wav_speech + alpha * wav_noise) * 32768
    noisy_wav = noisy_wav.astype(np.int16)

    return noisy_wav


def main():
    argvs = sys.argv[1:] # sys.argv는 실행 시의 parameter 값을 저장함. 단, index 0에는 실행하는 py코드명이 담김. index 1부터 parameter가 담기는 것임.
    if len(argvs) != 3: # parameter가 3개가 아닌 경우 에러. mode / snr / fs. 총 3개를 입력해줘야함.
        print('Error: Invalid input arguments')
        print('\t Usage: python generate_noisy_data.py [mode] [snr] [fs]')
        print("\t\t [mode]: 'train', 'validation'")
        print("\t\t [snr]: '0', '0, 5', ...'") # snr은 신호 대 잡음비로. 0이면 신호와 잡음 비율이 같다는 의미. 즉, 높을수록 잡음이 적어 좋은 것임.
        print("\t\t [fs]: '16000', ...")
        exit()
    mode = argvs[0] # mode
    snr_set = argvs[1].split(',') # snr / , 기준으로 잘라서 snr_set에 저장.
    fs = int(argvs[2]) # fs

    # Set speech and noise directory.
    speech_dir = Path("./Dataset")

    # Make a speech file list.
    speech_mode_clean_dir = speech_dir / mode / 'clean_aug' # pathlib를 통해 '/'를 경로 구분 문자로 사용할 수 있음.
    speech_mode_noisy_dir = speech_dir / mode / 'noisy_aug' # noisy 파일 저장 경로
    list_speech_files = scan_directory(speech_mode_clean_dir)

    # Make directories of the mode and noisy data.
    if os.path.isdir(speech_mode_clean_dir) is False:
        os.system('mkdir ' + str(speech_mode_clean_dir)) # os.system을 통해 CMD 명령어를 사용할 수 있음.

    if os.path.isdir(speech_mode_noisy_dir) is False:
        os.system('mkdir ' + str(speech_mode_noisy_dir))

    # Define a log file name.
    log_file_name = Path("./log_generate_data_" + mode + ".txt")
    f = open(log_file_name, 'w')

    if mode == 'train':
        # Make a noise file list
        noise_subset_dir = speech_dir / 'train' / 'noise'
        list_noise_files = scan_directory(noise_subset_dir) # train noise 디렉토리의 파일들을 리스트에 담음
        for snr_in_db in snr_set:
            for addr_speech in list_speech_files: # clean 디렉토리의 파일들을 불러옴
                # Load speech waveform and its sampling frequency.
                wav_speech, read_fs = soundfile.read(addr_speech) # wav_speech는 data / read_fs는 sample rate(여기서는 16000)를 의미함.
                if read_fs != fs:
                    wav_speech = librosa.resample(wav_speech, read_fs, fs) # clean파일의 sampe rate이 fs와 같지 않다면 fs(16000)으로 resample

                # Select a noise component randomly, and read it.
                nidx = np.random.randint(0, len(list_noise_files)) # noise 파일 중 랜덤으로 하나를 뽑음
                addr_noise = list_noise_files[nidx]  # 랜덤으로 뽑은 noise파일 주소를 addr_noise에 담음
                wav_noise, read_fs = soundfile.read(addr_noise) # 랜덤으로 뽑은 noise 파일의 data와 sample rate을 얻음
                if wav_noise.ndim > 1:
                    wav_noise = wav_noise.mean(axis=1)
                if read_fs != fs:
                    wav_noise = librosa.resample(wav_noise, read_fs, fs)

                # Generate noisy speech by mixing speech and noise components.
                # clean / noise / snr을 파라메터로 넣어 noisy 데이터를 얻음
                wav_noisy = generate_noisy_wav(wav_speech, wav_noise, int(snr_in_db))
                # noisy 데이터의 이름 = clean 이름 + noise 이름 + snr
                noisy_name = Path(addr_speech).name[:-4] +'_' + Path(addr_noise).name[:-4] + '_' + str(
                                  int(snr_in_db)) + '.wav'
                addr_noisy = speech_mode_noisy_dir / noisy_name # 생성한 noisy 파일은 mode/noisy  디렉토리에 담고 그 경로.
                wav.write(addr_noisy, fs, wav_noisy) # addr_noisy 경로에 wav_noisy 파일을 fs로 생성.

                # Display progress.
                print('%s > %s' % (addr_speech, addr_noisy))
                f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, int(snr_in_db)))

    elif mode == 'validation':
        # Make a noise file list for validation.
        noise_subset_dir = speech_dir / 'train' / 'noise' # noise 데이터는 train의 noise를 다시 사용함. 단, clean은 train때 사용하지 않은 것으로만 사용해야 함.!
        list_noise_files = scan_directory(noise_subset_dir)

        for addr_speech in list_speech_files: # clean 데이터는 처음에 인자로 준 mode에 따라 변경됨.
            # Load speech waveform and its sampling frequency.
            wav_speech, read_fs = soundfile.read(addr_speech)
            if read_fs != fs:
                wav_speech = librosa.resample(wav_speech, read_fs, fs)

            # Select a noise component randomly, and read it.
            nidx = np.random.randint(0, len(list_noise_files))
            addr_noise = list_noise_files[nidx]
            wav_noise, read_fs = soundfile.read(addr_noise)
            if wav_noise.ndim > 1:
                wav_noise = wav_noise.mean(axis=1)
            if read_fs != fs:
                wav_noise = librosa.resample(wav_noise, read_fs, fs)

            # Select an SNR randomly.
            # input한 SNR 중 랜덤으로 하나를 선택하는데, 그 이유는.. ?
            ridx_snr = np.random.randint(0, len(snr_set))
            snr_in_db = int(snr_set[ridx_snr])

            # Generate noisy speech by mixing speech and noise components.
            wav_noisy = generate_noisy_wav(wav_speech, wav_noise, snr_in_db)

            # Write the generated noisy speech into a file.
            noisy_name = Path(addr_speech).name[:-4] + '_' + Path(addr_noise).name[:-4] + '_' + str(
                              snr_in_db) + '.wav'
            addr_noisy = speech_mode_noisy_dir / noisy_name
            wav.write(addr_noisy, fs, wav_noisy)

            # Display progress.
            print('%s > %s' % (addr_speech, addr_noisy))
            f.write('%s\t%s\t%s\t%d dB\n' % (addr_noisy, addr_speech, addr_noise, snr_in_db))
    f.close()

if __name__ == '__main__':
    main()
