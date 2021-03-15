import itertools
import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import math
import time
import Synth_data as sd

from sklearn.model_selection import train_test_split

experiment_id = f"plots/VRAE/experiment-2-10.10"

if not os.path.isdir(experiment_id):
    os.mkdir(experiment_id)

def get_batch(samples, batch_idx, batch_size):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos]


def sine_wave(seq_length=30, num_samples=28 * 5 * 100, num_signals=1,
              freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9,
              random_seed=None, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)  # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)  # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A * np.sin(2 * np.pi * f * ix / float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples


samples = sine_wave()

inputs_train, vali_test = train_test_split(samples, test_size=0.4)
inputs_validation, inputs_test = train_test_split(vali_test, test_size=0.2)

sd.save_plot_sample(inputs_test[-6:], '0', 'real_data', path=experiment_id)

print("data loaded.")

batch_size = 32

print("data loaded.")

seq_length = inputs_train.shape[1]
num_features = inputs_train.shape[2]
# not used
random_seed = 0


########################
# ENCODER
########################

def encoder(hidden_units_enc, emb_dim, mult):
    with tf.variable_scope("encoder") as scope:
        input_seq_enc = tf.placeholder(tf.float32, [batch_size, seq_length, num_features])

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_enc, state_is_tuple=True)
        enc_rnn_outputs, enc_rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            # sequence_length=[seq_length]*batch_size,
            inputs=input_seq_enc)

        z_mean = tf.layers.dense(enc_rnn_states[1], emb_dim)
        z_log_sigma_sq = tf.layers.dense(enc_rnn_states[1], emb_dim)

        # Draw one sample z from Gaussian distribution with mean 0 and std 1
        eps = tf.random_normal((batch_size, emb_dim), 0, 1, dtype=tf.float32)

        # z = mu + sigma*epsilon
        latent_emb = tf.add(z_mean, tf.multiply(tf.exp(tf.multiply(z_log_sigma_sq, 0.5)), eps))

        latent_loss = mult * (-0.5) * tf.reduce_sum(1 + z_log_sigma_sq
                                                    - tf.square(z_mean)
                                                    - tf.exp(z_log_sigma_sq), 1)

        latent_loss = tf.reduce_mean(latent_loss)

        return input_seq_enc, enc_rnn_outputs, enc_rnn_states, latent_emb, latent_loss

########################
# DECODER
########################

def decoder(hidden_units_dec, latent_emb, input_seq_enc):
    with tf.variable_scope("decoder") as scope:
        W_out_dec = tf.Variable(tf.truncated_normal([hidden_units_dec, num_features]))
        b_out_dec = tf.Variable(tf.truncated_normal([num_features]))

        dec_inputs = tf.zeros(tf.shape(input_seq_enc))

        dec_initial_state = tf.layers.dense(latent_emb, hidden_units_dec, activation=tf.nn.tanh)


        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_units_dec)
        dec_rnn_outputs, dec_rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            # sequence_length=[seq_length]*batch_size,
            initial_state=dec_initial_state,
            inputs=dec_inputs)
        rnn_outputs_2d = tf.reshape(dec_rnn_outputs, [-1, hidden_units_dec])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_dec) + b_out_dec
        output_3d = tf.reshape(logits_2d, [-1, seq_length, num_features])
        # output_3d = tf.tanh(output_3d)
        reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(output_3d, input_seq_enc)))
        # reconstruction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_3d, labels=input_seq_enc))

    return reconstruction_loss, output_3d




learning_rate = 0.001
delta_error = 0
optimizer_str = "adam"
hidden_units_dec = 10
hidden_units_enc = 10
emb_dim = 10
mult = 0.1


# configs = itertools.product(learning_rate, delta_error, optimizer_str, hidden_units_dec, hidden_units_enc, emb_dim,
#                             mult)
config_keys = ['learning_rate', 'delta_error', 'optimizer_str', 'hidden_units_dec', 'hidden_units_enc', 'emb_dim',
               'mult']
verbose = 3

max_epochs = 200  # 2000
patience = 5
minibatch_size_train = batch_size
minibatch_size_validation = batch_size
minibatch_size_test = batch_size

_ = tf.Variable(initial_value=0)

sess = tf.Session()
saver = tf.train.Saver()

# for config in configs:

num_mini_batches_train = int(math.ceil(len(inputs_train) / float(minibatch_size_train)))
num_mini_batches_validation = int(math.ceil(len(inputs_validation) / float(minibatch_size_validation)))
num_mini_batches_test = int(math.ceil(len(inputs_test) / float(minibatch_size_test)))

experiment_random_id = str(int(np.random.rand(1) * 1000000))
# config_id = str(config).replace(", ", "_").replace("'", "")[1:-1] + "_" + experiment_random_id
# if verbose > 0:
    # print(config_id)

# learning_rate = config[config_keys.index('learning_rate')]
# delta_error = config[config_keys.index('delta_error')]
# optimizer_str = config[config_keys.index('optimizer_str')]

with tf.variable_scope("trainer"):

    # hidden_units_enc = config[config_keys.index('hidden_units_enc')]
    # hidden_units_dec = config[config_keys.index('hidden_units_dec')]
    # emb_dim = config[config_keys.index('emb_dim')]
    # mult = config[config_keys.index('mult')]

    input_seq_enc, enc_rnn_outputs, enc_rnn_states, latent_emb, latent_loss = encoder(hidden_units_enc, emb_dim,
                                                                                      mult)
    reconstruction_loss, output_3d_pred = decoder(hidden_units_dec, latent_emb, input_seq_enc)
    cost = reconstruction_loss + latent_loss

    # global_step = tf.Variable(np.int64(0), name='global_step', trainable=False)

    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)




sess.run(tf.global_variables_initializer())


# train
global_steps_count = 0
keep_training = True
epoch_counter = 1
patience_counter = 0
best_val_cost = 9999999999

costs_train = []
costs_val = []
costs_test = []
saved = False

while keep_training:

    current_time = time.time()



    time_start = time.time()
    # shuffle data
    np.random.shuffle(inputs_train)

    weighted_cost_sum = 0
    train_res = []
    for mbi in range(num_mini_batches_train):
        input_ = get_batch(inputs_train, mbi, minibatch_size_train)
        # FIX THIS! deal with last samples available in the set
        if len(input_) == batch_size:
            feed_dict = {input_seq_enc: input_}
            train_res = sess.run([train, cost, reconstruction_loss, latent_loss, output_3d_pred],
                                 feed_dict=feed_dict)
        # if config[2] == "adagrad_epochs":
        #     global_steps_count += 1


    weighted_cost_sum = 0
    test_res = []
    for mbi in range(num_mini_batches_test):
        input_ = get_batch(inputs_test, mbi, minibatch_size_test)
        if len(input_) == batch_size:
            feed_dict = {input_seq_enc: input_}
            test_res = sess.run([cost, output_3d_pred], feed_dict=feed_dict)
            weighted_cost_sum += test_res[0] * len(input_)

    sd.save_plot_sample(test_res[1][-6:], "epoch" + str(epoch_counter).zfill(4),
                     'test_VRAE', path=experiment_id)
    print(f"Epoch: {epoch_counter}")
    print(f" Time: {time.time() - current_time}")
    current_time = time.time()
    epoch_counter +=1

    if (epoch_counter > 200):
        break

saver.save(sess, 'Saved-sess/VRAE-experiment-2-10.10')









































