import os
import time
import torch
import numpy as np
import config as cfg
import itertools
from model import complex_model
from write_on_tensorboard import Writer
from dataloader import create_dataloader
from tester import model_test


###############################################################################
#                        Helper function definition                           #
###############################################################################
# Write training related parameters into the log file.
def write_status_to_log_file(fp, total_parameters):
    fp.write('%d-%d-%d %d:%d:%d\n' %
             (time.localtime().tm_year, time.localtime().tm_mon,
              time.localtime().tm_mday, time.localtime().tm_hour,
              time.localtime().tm_min, time.localtime().tm_sec))
    fp.write('total params   : %d (%.2f M, %.2f MBytes)\n' %
             (total_parameters,
              total_parameters / 1000000.0,
              total_parameters * 4.0 / 1000000.0))


# Calculate the size of total network.
def calculate_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


###############################################################################
#         Parameter Initialization and Setting for model training             #
###############################################################################
# Set device
if cfg.DEVICE == 'cuda':
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

if cfg.cycle:
    # Set model
    N2C = complex_model().to(DEVICE)
    C2N = complex_model().to(DEVICE)
    # Set optimizer and learning rate
    optimizer = torch.optim.Adam(itertools.chain(N2C.parameters(), C2N.parameters()), lr=cfg.learning_rate)
    total_params = calculate_total_params(N2C) + calculate_total_params(C2N)
else:
    # Set model
    model = complex_model().to(DEVICE)
    # Set optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    total_params = calculate_total_params(model)

if cfg.masking_mode == 'Direct(None make)':
    direct = True
else:
    direct = False
###############################################################################
#                        Confirm model information                            #
###############################################################################
print('%d-%d-%d %d:%d:%d\n' %
      (time.localtime().tm_year, time.localtime().tm_mon,
       time.localtime().tm_mday, time.localtime().tm_hour,
       time.localtime().tm_min, time.localtime().tm_sec))
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

###############################################################################
#                        Set a log file to store progress.                    #
#               Set a hps file to store hyper-parameters information.         #
###############################################################################
if cfg.chkpt_model is not None:  # Load the checkpoint
    print('Resuming from checkpoint: %s' % cfg.chkpt_path)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.chkpt_model
    dir_to_logs = cfg.logs_dir + cfg.chkpt_model

    if cfg.cycle:
        N2C_checkpoint = torch.load(dir_to_save + 'N2C_chkpt_' + cfg.chkpt + '.pt')
        N2C.load_state_dict(N2C_checkpoint['model'])
        optimizer.load_state_dict(N2C_checkpoint['optimizer'])
        epoch_start_idx = N2C_checkpoint['epoch'] + 1

        model = N2C
    else:
        checkpoint = torch.load(cfg.chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start_idx = checkpoint['epoch'] + 1
    mse_vali_total = np.load(str(dir_to_save + '/mse_vali_total.npy'))
    # if the loaded length is shorter than I expected, extend the length
    if len(mse_vali_total) < cfg.max_epochs:
        plus = cfg.max_epochs - len(mse_vali_total)
        mse_vali_total = np.concatenate((mse_vali_total, np.zeros(plus)), 0)
else:
    print('File load error: there is no check point!')


# logging
log_fname = str(dir_to_save + '/log.txt')
if not os.path.exists(log_fname):
    fp = open(log_fname, 'w')
    write_status_to_log_file(fp, total_params)
else:
    fp = open(log_fname, 'a')

###############################################################################
###############################################################################
#                             Main program start !!                           #
###############################################################################
###############################################################################
# Writer initialize
writer = Writer(dir_to_logs)

###############################################################################
#                                    Test                                     #
###############################################################################
print('Starting test run')
noise_type = ['seen', 'unseen']
noisy_snr = ['0', '5', '10', '15', '20']

# Road the dataset information to compare
# data_info = np.load('./input/C1_dataset_info.npy')
# data_info[0] = data_info[0][0]

for type in range(len(noise_type)):
    for snr in range(1, len(noisy_snr)):

        test_loader = create_dataloader(mode='test', type=type, snr=snr)
        test_pesq, test_stoi, test_csig, test_cbak, test_cvol = model_test(
            model, test_loader, noise_type[type], noisy_snr[snr], dir_to_save, direct, cfg.chkpt, DEVICE)

        # Road the score to compare
        # noisy_pesq = data_info[type][snr][0]
        # noisy_stoi = data_info[type][snr][1]
        # noisy_csig = data_info[type][snr][2]
        # noisy_cbak = data_info[type][snr][3]
        # noisy_cvol = data_info[type][snr][4]
        noisy_pesq = 0
        noisy_stoi = 0
        noisy_csig = 0
        noisy_cbak = 0
        noisy_cvol = 0

        print('Noise type {} | SNR {}'.format(noise_type[type], noisy_snr[snr]))
        fp.write('\n\nNoise type {} | SNR {}'.format(noise_type[type], noisy_snr[snr]))
        print('PESQ: REF {:.6} EST {:.6} | REF {:.6} EST STOI {:.6}'
              .format(noisy_pesq, test_pesq, noisy_stoi, test_stoi))
        print('REF CSIG {:.6f} | CBAK {:.6f} | COVL {:.6f}'.format(noisy_csig, noisy_cbak, noisy_cvol))
        print('    CSIG {:.6f} | CBAK {:.6f} | COVL {:.6f}'.format(test_csig, test_cbak, test_cvol))
        fp.write('PESQ: REF {:.6} EST {:.6} | REF {:.6} EST STOI {:.6f}'
                 .format(noisy_pesq, test_pesq, noisy_stoi, test_stoi))

# from scipy.io.wavfile import write as wav_write
#
# samples = np.load('./input/PAM_C1+validation_dataset.npy')
# sample_num = 4
# noisy_speech = samples[sample_num][0]
# clean_speech = samples[sample_num][1][0]
#
# inputs = torch.from_numpy(noisy_speech)
# inputs = inputs.float().to(DEVICE)
# inputs = inputs.unsqueeze(0)
#
# _, _, outputs = model(inputs, direct_mapping=direct)
#
# # estimate the output speech with pesq and stoi
# estimated_wavs = outputs.detach().numpy()
#
# wav_write('./models/sample_'+str(sample_num)+'_est_MSE.wav', 16000, estimated_wavs[0])
# # wav_write('./models/sample_'+str(sample_num)+'_clean.wav', 16000, clean_speech)
# # wav_write('./models/sample_'+str(sample_num)+'_noisy.wav', 16000, noisy_speech)
