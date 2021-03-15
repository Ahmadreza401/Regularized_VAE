from Offline_Disriminator import Off_Discrim
from Synth_data import random_input
import pickle
import matplotlib.pyplot as plt
import numpy as np

Training = False
TSTR = True

config = [70, 50, 20, 10]
batch_size = 100
seq_length = 144

tipe = 'Wind-Speed'
model = 'RGAN'
sampeling = 'post'
# trained_epoch = '-1000'
trained_epoch = ''

round = '1'
# train_with = 'other_data'
# train_with = 'early_epochs'
train_with = 'relative'
train_size = 15000

'''Train Model'''
if Training:

    # Loading The Real Data-set
    # real_vector = random_input(seq_length=100, num_samples=12000)
    if model == 'RGAN':
        f = open('Saved-vecs/' + tipe + '/02-' + model + trained_epoch + '.pckl', 'rb')
        real_vector = pickle.load(f)
        f.close()

    elif model == 'DVAE' or model == 'RVAE':
        f = open('Saved-vecs/' + tipe + '/02-' + model + trained_epoch + '-' + sampeling + '.pckl', 'rb')
        real_vector = pickle.load(f)
        choices = np.random.choice(len(real_vector), size=train_size, replace=False)
        real_vector = real_vector[choices]
        f.close()

    # Loading the Noise Data-set
    if train_with == 'early_epochs':
        f = open('Saved-vecs/' + tipe + '/02-noise-pre' + '.pckl', 'rb')
        noise_vector = pickle.load(f)
        f.close()
    elif train_with == 'relative':
        f = open('Saved-vecs/' + tipe + '/02-noise-rel-' + model + '.pckl', 'rb')
        noise_vector = pickle.load(f)
        f.close()

    # choices = np.random.choice(a=4000, size=20)

    # for choice in choices:
    #     plt.plot(real_vector[choice], label='real')
    #
    #     if train_with == 'early_epochs':
    #         plt.plot(noise_vector[1][choice], label='1')
    #         plt.plot(noise_vector[1][4000 + choice], label='1')
    #         plt.plot(noise_vector[1][8000 + choice], label='1')
    #
    #         plt.plot(noise_vector[11][choice], label='11')
    #         plt.plot(noise_vector[11][4000 + choice], label='11')
    #         plt.plot(noise_vector[11][8000 + choice], label='11')
    #
    #         plt.plot(noise_vector[21][choice], label='21')
    #         plt.plot(noise_vector[21][4000 + choice], label='21')
    #         plt.plot(noise_vector[21][8000 + choice], label='21')
    #
    #     elif train_with == 'relative':
    #         plt.plot(noise_vector[1][choice], label='1')
    #         plt.plot(noise_vector[1][4000 + choice], label='1')
    #         plt.plot(noise_vector[1][8000 + choice], label='2')
    #         plt.plot(noise_vector[1][10000 + choice], label='2')
    #
    #     plt.legend()
    #     plt.show()

    D = Off_Discrim(config=config, batch_size=batch_size, seq_length=seq_length)

    D.fit(train_data=real_vector, train_size=train_size, trained_epoch=trained_epoch, noise_data=noise_vector, train_with=train_with, tyype=tipe,
          trained_on=model, round=round, path_save='Saved-vecs/' + tipe + '/sv/' + model + '/')

elif TSTR:

    path_save = 'Saved-vecs/' + tipe + '/sv/' + model + '/'

    # Restoring the model trained on Synthetic
    D = Off_Discrim(config=config, batch_size=batch_size, seq_length=seq_length)
    D.saver.restore(D.sess, path_save + model + trained_epoch + '-' + train_with + '-' + str(train_size) + '-' + round)

    if tipe != 'Periodic':
        f = open('Saved-vecs/' + tipe + '/all-data' + '.pckl', 'rb')
        all_real_samples = pickle.load(f)
        f.close()

    percentages = []
    print('Truly Classified, Average, Stdev')
    for i in range(20):
        # Sampeling from Real Data-Set
        # real_samples = random_input(seq_length=100, num_samples=5000)

        if tipe == 'Periodic':
            real_samples = random_input(seq_length=seq_length, num_samples=5000)

        else:
            choices = np.random.choice(len(all_real_samples), size=5000, replace=False)
            real_samples = all_real_samples[choices]

        real_samples = real_samples.reshape([-1, seq_length, 1])

        # Computing TSTR
        p_real = D.sess.run(fetches=D.prob_tf, feed_dict={D.test_vector: real_samples})

        average_p = np.average(p_real)
        std_p = np.std(p_real)

        truly_categorized = [p for p in p_real if p > 0.5]

        print(f'{len(truly_categorized)/5000}, {average_p}, {std_p}')


    print('yes')






else:

    # path_save = 'Saved-vecs/01-Periodic/sv/'
    # trained_on = 'RGAN'
    #
    # # dvae_noise = fetch_VAE(noisy=False, model_name='DVAE', size=5000, seqLen=100, saved_epoch=100)
    # # tf.reset_default_graph()
    # #
    # # rvae_noise = fetch_VAE(noisy=False, model_name='RVAE', size=5000, seqLen=100, saved_epoch=100)
    # # tf.reset_default_graph()
    # #
    # # rgan_noise = fetch_RGAN(noisy=False, size=5000, saved_epoch=100)
    # # tf.reset_default_graph()
    #
    # real_samples = random_input(seq_length=100, num_samples=5000)
    #
    # D = Off_Discrim(config=config, D=d, batch_size=batch_size, seq_length=seq_length)
    # D.saver.restore(D.sess, path_save + trained_on)
    #
    # # p_dave = D.sess.run(fetches=D.prob_tf, feed_dict={D.test_vector: dvae_noise})
    # # p_rvae = D.sess.run(fetches=D.prob_tf, feed_dict={D.test_vector: rvae_noise})
    # # p_rgan = D.sess.run(fetches=D.prob_tf, feed_dict={D.test_vector: rgan_noise})
    #
    # p_real = D.sess.run(fetches=D.prob_tf, feed_dict={D.test_vector: real_samples})
    #
    # # n_dvae = n_rvae = n_rgan = 0
    # #
    # # for i in range(5000):
    # #     if p_dave[i] < 0.7:
    # #         n_dvae += 1
    # #     if p_rvae[i] < 0.7:
    # #         n_dvae += 1
    # #     if p_rgan[i] < 0.7:
    # #         n_rgan += 1
    # #
    # # print(f'Out Classified for DVAE: {n_dvae}, for RVAE: {n_rvae}, for RGAN: {n_rgan}')
    # #
    # #
    # # for i in range(10):
    # #     plt.plot(dvae_noise[i], color = 'blue')
    # #     plt.plot(rvae_noise[i], color = 'orange')
    # #     plt.plot(rgan_noise[i], color = 'red')
    # #     plt.plot(real_samples[i], color = 'green')
    # #
    # #     plt.show()


    print('yes')










