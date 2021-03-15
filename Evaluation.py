import pickle
import matplotlib.pyplot as plt
import numpy as np
from Evaluate import mmd
from Synth_data import random_input
import json

# Load Experiments Data

# Loading RGAN Samples
f = open('Saved-vecs/Solar/01-RGAN-Solar' + '.pckl', 'rb')
RGAN_1= pickle.load(f)
RGAN_1 = np.array(RGAN_1)
f.close()

# f = open('Saved-vecs/Smooth/01-RGAN-2' + '.pckl', 'rb')
# RGAN_2 = pickle.load(f)
# RGAN_2 = np.array(RGAN_2)
# f.close()

# Loading \betaRVAE Samples
f = open('Saved-vecs/Solar/01-RVAE-1-[100]--dim_z,3-beta,2-mult,0.0005' + '.pckl', 'rb')
RVAE = pickle.load(f)
f.close()

# f = open('Saved-vecs/Cnsm/01-RVAE-2-Active-0006-[100]--dim_z,3-beta,2-mult,0.0005' + '.pckl', 'rb')
# RVAE_2 = pickle.load(f)
# f.close()

# Loading \betaDVAE Samples
f = open('Saved-vecs/Solar/01-DVAE-1-[70, 50, 20, 10]--dim_z,3-beta,2-mult,0.0005' + '.pckl', 'rb')
DVAE_1 = pickle.load(f)
f.close()

# f = open('Saved-vecs/Smooth/01-DVAE-14-0028-[70, 50, 20, 10]--dim_z,3-beta,2.1-mult,0.0007' + '.pckl', 'rb')
# DVAE_2 = pickle.load(f)
# f.close()
#
# f = open('Saved-vecs/Smooth/01-DVAE-15-0030-[70, 50, 20, 10]--dim_z,3-beta,2.5-mult,0.0007' + '.pckl', 'rb')
# DVAE_3 = pickle.load(f)
# f.close()

# f = open('Saved-vecs/Smooth/01-DVAE-12-0022-[70, 50, 20, 10]--dim_z,3-beta,1.9-mult,0.0006' + '.pckl', 'rb')
# DVAE_4 = pickle.load(f)
# f.close()



mmd_RGAN_1 = []
mmd_RGAN_2 = []

mmd_RVAE = []
mmd_RVAE_2 = []

mmd_DVAE_1 = []
mmd_DVAE_2 = []
mmd_DVAE_3 = []
mmd_DVAE_4 = []

# eval_real = random_input(seq_length=100, num_samples=5000)

f = open('Saved-vecs/Solar/01-solar-all' + '.pckl', 'rb')
eval_real = pickle.load(f)
f.close()

eval_real = eval_real.reshape([-1, 144, 1])

choices = np.random.choice(a=len(eval_real) - 1, size=20000, replace=False)

for i in range(4):

    eval_real_temp = eval_real[choices[i*5000:(i+1)*5000]]
    # eval_real = np.float32(eval_real)

    mmd_RGAN_1_mid = mmd(generated_sampels=RGAN_1, real_samples=eval_real_temp)[0]
    # mmd_RGAN_2_mid = mmd(generated_sampels=RGAN_2, real_samples=eval_real)[0]

    mmd_RVAE_mid = mmd(generated_sampels=RVAE, real_samples=eval_real_temp)[0]
    # mmd_RVAE_2_mid = mmd(generated_sampels=RVAE_2, real_samples=eval_real)[0]


    mmd_DVAE_1_mid = mmd(generated_sampels=DVAE_1, real_samples=eval_real_temp)[0]
    # mmd_DVAE_2_mid = mmd(generated_sampels=DVAE_2, real_samples=eval_real)[0]
    # mmd_DVAE_3_mid = mmd(generated_sampels=DVAE_3, real_samples=eval_real)[0]
    # mmd_DVAE_4_mid = mmd(generated_sampels=DVAE_4, real_samples=eval_real)[0]

    # print("Run--" + str(i + 1) + ", RGAN_1: " + str(mmd_RGAN_1_mid) + ", RGAN_2: " + str(mmd_RGAN_2_mid)
    #       + ", RVAE: " + str(mmd_RVAE_mid)
    #       + ", DVAE_1: " + str(mmd_DVAE_1_mid) + ", DVAE_2: " + str(mmd_DVAE_2_mid)
    #       + ", DVAE_3: " + str(mmd_DVAE_3_mid) + ", DVAE_4: " + str(mmd_DVAE_4_mid))

    print("Run--" + str(i + 1)
          # + ", DVAE_13: " + str(mmd_DVAE_1_mid) )#
          + ", RGAN_1: " + str(mmd_RGAN_1_mid) + ", RVAE_1: " + str(mmd_RVAE_mid) +
          # ", RVAE_2: " + str(mmd_RVAE_2_mid)
          ", DVAE_1: " + str(mmd_DVAE_1_mid))
          # + ", DVAE_15: " + str(mmd_DVAE_3_mid) ) # + ", DVAE_12: " + str(mmd_DVAE_4_mid))

#     mmd_RGAN_1.append(mmd_RGAN_1_mid)
#     mmd_RGAN_2.append(mmd_RGAN_2_mid)
#
#     mmd_RVAE.append(mmd_RVAE_mid)
#
#     mmd_DVAE_1.append(mmd_DVAE_1_mid)
#     mmd_DVAE_2.append(mmd_DVAE_2_mid)
#     mmd_DVAE_3.append(mmd_DVAE_3_mid)
#     mmd_DVAE_4.append(mmd_DVAE_4_mid)
#
#
# eval_mmd_RGAN = np.array(mmd_RGAN).mean()
#
# eval_mmd_RVAE = np.array(mmd_RVAE).mean()
#
# eval_mmd_DVAE = np.array(mmd_DVAE).mean()
#
#
# to_store = {'eval_mmd_RGAN': eval_mmd_RGAN, 'eval_mmd_RVAE': eval_mmd_RVAE,
#             'eval_mmd_DVAE': eval_mmd_DVAE}
# with open("Saved-vecs/Sine/1-Comparison.json", "w") as f:
#     json.dump(to_store, f)
#
#
# print("mmd RGAN: " + str(eval_mmd_RGAN) + ", mmd RVAE: " + str(eval_mmd_RVAE) + ", mmd DVAE: "
#       + str(eval_mmd_DVAE))





























