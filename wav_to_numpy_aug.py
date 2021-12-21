from generate_noisy_data import scan_directory
import soundfile
import numpy as np
import time
list_noisy_files = scan_directory('./Dataset/train/noisy_aug')

def normalization(dataset):
    for i in range(len(dataset)):
        noisy_max = np.max(abs(dataset[i][0]))
        dataset[i][0] = dataset[i][0] / (noisy_max + 1e-7)

        clean_max = np.max(abs(dataset[i][1]))
        dataset[i][1] = dataset[i][1] / (clean_max + 1e-7)
    return dataset

dataset = []
batch = 0
for noisy_addr in (list_noisy_files):
    st_time = time.time()
    batch += 1
    data_noisy, fs_noisy = soundfile.read(noisy_addr)
    clean_addr = str(noisy_addr).replace('noisy', 'clean')

    _lst = [] # '_'의 인덱스를 담을 리스트
    for idx, char in enumerate(clean_addr):
        if (char == '_'):
            _lst.append(idx) # '_'의 index를 추가함

    clean_addr = clean_addr[:_lst[9]] # clean 파일에 해당하는 name까지만 자름
    clean_addr += '.wav'

    data_clean, fs_clean = soundfile.read(clean_addr) # clean 데이터 읽기
    dataset.append([data_noisy, data_clean]) # clean / noisy 데이터 리스트에 추가
    print('{} done... takes {:.4} seconds...'.format(batch, time.time()-st_time))
train_dataset = np.array(dataset)
train_dataset = normalization(train_dataset)

np.save('./Dataset/train_dataset_norm_tv31_snr51015_minus.npy',train_dataset)

# load_data = np.load('./Dataset/train_dataset_norm.npy')
# print(load_data.shape)
