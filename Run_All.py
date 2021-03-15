import numpy as np
import matplotlib.pyplot as plt
import os
import Synth_data as sd
import pickle
import Dense_all as dl


seqLen = 100

'''Periodic Experiments'''
# time_series, _, _, _, _ = sd.random_input(seq_length=seqLen)
time_series, _, _, _, _ = sd.continuous_input(seq_length=seqLen)

'''Loading Smooth input'''
# f = open('Saved-vecs/Smooth/all-data' + '.pckl', 'rb') # input size is 100
# time_series = pickle.load(f)
# time_series = time_series[:30000]
# f.close()

'''Loading Household Power Consumption data set'''
# f = open('Saved-vecs/Cnsm/all-data' + '.pckl', 'rb')  # input size is 60
# time_series = pickle.load(f)
# time_series = time_series[:10200]
# f.close()

'''Loading Wind_Power input'''
# f = open('Saved-vecs/Wind-Power/all-data' + '.pckl', 'rb')  # input size is 144
# time_series = pickle.load(f)
# time_series = time_series[:30000]
# f.close()

'''Loading Wind_Speed input'''
# f = open('Saved-vecs/Wind-Speed/all-data' + '.pckl', 'rb')  # input size is 144
# time_series = pickle.load(f)
# time_series = time_series[:30000]
# f.close()

'''Loading Solar data set'''
# f = open('Saved-vecs/Solar/all-data' + '.pckl', 'rb')  # input size is 144
# time_series = pickle.load(f)
# time_series = time_series[:30000]
# f.close()

# time_test_specific = sd.generate_specific_sample(seq_len=seqLen, A=0.32, f=3, offset=0)

# plt.plot(time_test_specific)
# plt.show()


# configs_3l = [[50, 20, 10]#,
              # [70, 30, 10]
             # ]
#
configs_4l = [[70, 50, 20, 10]#,
#               [80, 60, 30, 15]
             ]

# configs_5l = [[90, 70, 50, 30, 10]]

input_type = 'Periodic'

path_experiment = 'experiments/Dense/' + str(3) + '-' + input_type + '-elbo1'
path_save = 'Saved-sess/Dense/' + str(3) + '-' + input_type + '-elbo1'

if not os.path.exists(path_experiment):
    os.mkdir(path_experiment)

if not os.path.exists(path_save):
    os.mkdir(path_save)

# mults = [0.1, 0.01, 0.001]
mults = [0.0005]
# betas = [np.round(l, 2) for l in np.linspace(start=1, stop=6, num=26)]
# betas = [1, 2, 3]
# betas = [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3]
# betas = [0, 0.00001, 0.00003, 0.00005, 0.00007, 0.00009, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009]
betas = [2]
# dims = [d for d in range(3, 11)]
# dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 40, 50, 75, 100]
dims = [3]

exp_number = 1
best_elbo = -9999999999999999
done = 3
patience = 0
last_elbo = 0


for confg in configs_4l:

    for dim_z in dims:

        for mult in mults:

            for beta in betas:

                vae = dl.TimeVAE(config=confg, D=[seqLen, 1], dim_z=dim_z,
                                 beta=beta, mult=mult)
                last_elbo, early_termination = vae.fit(time_series, input_type, exp_number, path_experiment, path_save)


                exp_number += 1

                # if last_elbo < best_elbo:
                #     patience = 0
                #     best_elbo = last_elbo
                # else:
                #     patience += 1

                if early_termination:
                    patience += 1
                else:
                    patience = 0

                if patience > done:
                    early_termination = False
                    patience = 0
                    break





































