from Synth_data import random_input
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Evaluate import mmd

tipe = 'Solar'
sampeling = 'post'
seq_length = 144

rgan_epoch = ''
rvae_epoch = ''
dvae_epoch = ''

end_point = 5000
start_point = 35000

if tipe == 'Cnsm':
    end_point = 2000
    start_point = 12200

path_load = 'Saved-vecs/' + tipe + '/02-'

f = open(path_load + 'RGAN' + '.pckl', 'rb')
RGAN_samples = pickle.load(f)[:end_point]
f.close()

f = open(path_load + 'RVAE-' + sampeling + '.pckl', 'rb')
RVAE_samples = pickle.load(f)[:end_point]
f.close()

f = open(path_load + 'DVAE-' + sampeling + '.pckl', 'rb')
DVAE_samples = pickle.load(f)[:end_point]
f.close()

# empirical_dist = random_input(seq_length=100, num_samples=20000)
f = open('Saved-vecs/' + tipe + '/all-data' + '.pckl', 'rb')
empirical_dist = pickle.load(f)[start_point:]
f.close()

empirical_dist = np.reshape(empirical_dist, [-1, seq_length, 1])
empirical_dist = np.float32(empirical_dist)

for i in range(10):

    plt.plot(RGAN_samples[i], color='red', label='RGAN')
    plt.plot(RVAE_samples[i], color='orange', label='RVAE')
    plt.plot(DVAE_samples[i], color='blue', label='DVAE')
    plt.plot(empirical_dist[i], color='green')

    plt.legend()
    plt.show()

rounds = len(empirical_dist)//end_point
for i in range(4):

    dist_temp = empirical_dist[i * end_point:(i + 1) * end_point]

    mmd_RGAN = mmd(generated_sampels=RGAN_samples, real_samples=dist_temp, seqLength=seq_length)[0]

    mmd_RVAE = mmd(generated_sampels=RVAE_samples, real_samples=dist_temp, seqLength=seq_length)[0]

    mmd_DVAE = mmd(generated_sampels=DVAE_samples, real_samples=dist_temp, seqLength=seq_length)[0]

    print("Run--" + str(i + 1)
          + ", RGAN_1: " + str(mmd_RGAN) + ", RVAE_1: " + str(mmd_RVAE) + ", DVAE_1: " + str(mmd_DVAE))




