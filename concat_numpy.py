# npy 파일 이어 붙이기
import numpy as np

a1 = np.load('./Dataset/train_shifting+minus+ori_data.npy')
a2 = np.load('./Dataset/train_dataset_norm_tv31_snr51015_reverse.npy')

arr = np.concatenate((a1, a2))
print(len(arr))

np.save('./Dataset/train_shifting+minus+reverse+ori_data.npy', arr)