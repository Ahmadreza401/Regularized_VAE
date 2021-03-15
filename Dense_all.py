import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_probability as tfp
import functools
import time
import os

import Synth_data  as sd

import json

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import functools

class TimeVAE:
    def make_encoder(self, config, data, code_size):

        # dense_enc = functools.partial(tf.keras.layers.Dense, activation=tf.nn.relu)
        # layerz_enc = tf.get_variable(name='layers_dec', shape=[len(config)],
        #                              dtype=tf.variant, initializer=tf.initializers.he_normal())
        #
        # for confe in config:
        #     layerz_enc.append(dense_enc(confe))
        #
        # encoderr = tf.get_variable(name='encdr', shape=[1], dtype=tf.variant,
        #                            initializer=tf.initializers.glorot_normal())
        #
        # encoderr = tf.keras.Sequential(layerz_enc)
        #
        # x = tf.layers.flatten(data)
        # x = encoderr(x)

        '''Old version of implementing encoder'''

        x = tf.layers.flatten(data)
        x = tf.layers.dense(x, config[0], tf.nn.relu)
        x = tf.layers.dense(x, config[1], tf.nn.relu)
        x = tf.layers.dense(x, config[2], tf.nn.relu)
        x = tf.layers.dense(x, config[3], tf.nn.relu)
        # x = tf.layers.dense(x, config[4], tf.nn.relu)
        # x = tf.layers.dense(x, config[5], tf.nn.relu)

        self.z_mean = loc = tf.layers.dense(x, code_size)
        self.z_log_sigma_sq = scale = tf.layers.dense(x, code_size, tf.nn.softplus)
        # return tfd.MultivariateNormalDiag(loc, scale)

        return tfp.distributions.MultivariateNormalDiag(loc, scale)

    def make_prior(self, code_size):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        # return tfd.MultivariateNormalDiag(loc, scale)
        return tfp.distributions.MultivariateNormalDiag(loc, scale)

    def make_decoder(self, config, code, data_shape):


        # x = code
        #
        # dense_dec = functools.partial(tf.keras.layers.Dense, activation=tf.nn.relu)
        # layerz_dec = []
        #
        # # config.reverse()
        #
        # for confd in confg:
        #     layerz_dec.append(dense_dec(confd))
        #
        # decoderr = tf.keras.Sequential(layerz_dec)
        #
        # x = decoderr(x)


        '''Old way of implementing decoder'''

        # config.reverse()

        x = code
        x = tf.layers.dense(x, config[0], tf.nn.relu)
        x = tf.layers.dense(x, config[1], tf.nn.relu)
        x = tf.layers.dense(x, config[2], tf.nn.relu)
        x = tf.layers.dense(x, config[3], tf.nn.relu)
        # x = tf.layers.dense(x, config[4], tf.nn.relu)
        # x = tf.layers.dense(x, config[5], tf.nn.relu)

        logit = tf.layers.dense(x, np.prod(data_shape))
        self.gen_output = logit = tf.reshape(logit, [-1] + data_shape)
            # return tfd.Independent(tfd.Bernoulli(logit), 2)
        return tfp.distributions.Independent(tfp.distributions.Bernoulli(logit))

    def __init__(self, config, D, dim_z, beta, mult):

        self.beta = beta
        self.dim_z = dim_z
        self.config = config
        self.D = D
        self.mult = mult

        config_rev = []
        for conf in reversed(config):
            config_rev.append(conf)

        self.encoder = tf.make_template('encoder', self.make_encoder)
        self.decoder = tf.make_template('decoder', self.make_decoder)

        self.data = tf.placeholder(tf.float32, shape=[None] + D)

        self.prior = self.make_prior(code_size=self.dim_z)
        self.posterior = self.encoder(config, self.data, code_size=self.dim_z)
        self.code = self.posterior.sample()

        self.encoder(config, self.data, code_size=self.dim_z)


        self.likelihood = self.decoder(config_rev, self.code, D).log_prob(self.data)

        self.divergence = tfp.distributions.kl_divergence(self.posterior, self.prior)

        self.latent_loss = (-0.5) * tf.reduce_sum(1 + self.z_log_sigma_sq
                                                    - tf.square(self.z_mean)
                                                    - tf.exp(self.z_log_sigma_sq), 1)

        self.recon_loss = tf.reduce_mean(tf.square(tf.subtract(self.gen_output, self.data)))

        self.cost = tf.reduce_mean(self.recon_loss + self.beta * self.latent_loss)

        self.elbo = tf.reduce_mean(self.likelihood - self.beta * self.divergence)
        self.elbo1 = tf.reduce_mean(self.likelihood - self.beta * self.latent_loss)

        self.optimize = tf.train.AdamOptimizer(0.001).minimize(-self.elbo1)

        self.tf_code = tf.placeholder(dtype=tf.float32, shape=[None, self.dim_z])
        # self.test_decoded = self.decoder(self.tf_code, [28, 28]).mean()
        # self.test_decoded = self.decoder(self.tf_code, [10, 10]).mean()
        self.test_decoded = self.decoder(config_rev, self.tf_code, D).mean()

        # self.samples = self.decoder(self.prior.sample(10), [28, 28]).mean()
        # self.samples = self.decoder(self.prior.sample(10), [10, 10]).mean()
        self.samples = self.decoder(config_rev, self.prior.sample(10), D).mean()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)


    def fit(self, X, input_type, exp_number, path_exp, path_save, epochs= 500, batch_sz=100):

        X = X.reshape([len(X)] + self.D)

        train_set, vali_test = train_test_split(X, test_size=0.4)
        vali_set, test_set = train_test_split(vali_test, test_size=0.2)

        n_batches_train = len(train_set) // batch_sz
        n_batches_vali = len(vali_set) // batch_sz
        n_batches_test = len(test_set) // batch_sz

        # train_set = train_set.reshape([len(train_set)] + self.D)
        # vali_set = vali_set.reshape([len(vali_set)] + self.D)
        # test_set = test_set.reshape([len(test_set)] + self.D)

        patience = 20
        max_epochs = 300
        delta_error = 0
        experiment_number = exp_number

        path_experiment = path_exp

        costs_train = []
        costs_vali = []
        costs_test = []

        elbos_train = []
        elbos_vali = []
        elbos_test = []

        elbos1_train = []
        elbos1_vali = []
        elbos1_test = []

        keep_training = True

        patience_cost = 0
        patience_elbo = 0
        patience_elbo1 = 0

        epoch_counter = 0

        best_cost_vali = 99999999999999
        best_elbo_vali = -99999999999999
        best_elbo1_vali = -9999999999999

        early_termination = False

        no_cost_improve = False
        no_elbo_improve = False
        no_elbo1_improve = False

        epoch_falg = 50
        time_epcoh = time.time()

        start_time = time.time()

        '''Print Experiment Information'''
        print('Running ' + input_type + ' ' + str(self.D))
        print('Epoch, Elbo1_vali, Epoch Time')

        while keep_training:

            '''Train'''
            weighted_cost_train = 0
            weighted_elbo_train = 0
            weighted_elbo1_train = 0
            for i in range(n_batches_train):
                batch = train_set[i * batch_sz:(i+1) * batch_sz]

                _, cost_cur, elbo_cur, elbo1_cur = self.sess.run(fetches=[self.optimize, self.cost, self.elbo, self.elbo1],
                                                                 feed_dict={self.data: batch})
                weighted_cost_train += cost_cur * len(batch)
                weighted_elbo_train += elbo_cur * len(batch)
                weighted_elbo1_train += elbo1_cur * len(batch)

            cost_train = weighted_cost_train / len(train_set)
            costs_train.append(cost_train)

            elbo_train = weighted_elbo_train / len(train_set)
            elbos_train.append(elbo_train)

            elbo1_train = weighted_elbo1_train / len(train_set)
            elbos1_train.append(elbo1_train)


            '''Vali'''
            weighted_cost_vali = 0
            weighted_elbo_vali = 0
            weighted_elbo1_vali = 0
            for i in range(n_batches_vali):
                batch = vali_set[i * batch_sz:(i + 1) * batch_sz]

                _, cost_cur, elbo_cur, elbo1_cur = self.sess.run(
                    fetches=[self.optimize, self.cost, self.elbo, self.elbo1],
                    feed_dict={self.data: batch})
                weighted_cost_vali += cost_cur * len(batch)
                weighted_elbo_vali += elbo_cur * len(batch)
                weighted_elbo1_vali += elbo1_cur * len(batch)

            cost_vali = weighted_cost_vali / len(vali_set)
            costs_vali.append(cost_vali)

            elbo_vali = weighted_elbo_vali / len(vali_set)
            elbos_vali.append(elbo_vali)

            elbo1_vali = weighted_elbo1_vali / len(vali_set)
            elbos1_vali.append(elbo1_vali)

            # Checking whether the cost is improving or not
            if cost_vali <= best_cost_vali - delta_error:
                best_cost_vali = cost_vali
                patience_cost = 0
            else:
                patience_cost += 1
                if patience_cost > patience:
                    no_cost_improve = True

            # Checking whether the elbo is improving or not
            if elbo_vali >= best_elbo_vali + delta_error:
                best_elbo_vali = elbo_vali
                patience_elbo = 0
            else:
                patience_elbo += 1
                if patience_elbo > patience:
                    no_elbo_improve = True

            # Checking whether the elbo1 is improving or not
            if elbo1_vali >= best_elbo1_vali + delta_error:
                best_elbo1_vali = elbo1_vali
                patience_elbo1 = 0
            else:
                patience_elbo1 += 1
                if patience_elbo1 > patience:
                    no_elbo1_improve = True

            # Checking whether to continue training or to quit
            if no_elbo1_improve:
                # keep_training = False
                if epoch_counter < 30:
                    early_termination = True

            if epoch_counter > max_epochs:
                keep_training = False


            '''Test'''
            weighted_cost_test = 0
            weighted_elbo_test = 0
            weighted_elbo1_test = 0

            last_test_batch = []
            last_test_gen = []

            for i in range(n_batches_test):
                batch = test_set[i * batch_sz:(i + 1) * batch_sz]
                last_test_batch = batch

                _, cost_cur, elbo_cur, elbo1_cur, last_test_gen = self.sess.run(fetches=[self.optimize, self.cost, self.elbo, self.elbo1,
                                                                                      self.samples],
                                                                             feed_dict={self.data: batch})
                weighted_cost_test += cost_cur * len(batch)
                weighted_elbo_test += elbo_cur * len(batch)
                weighted_elbo1_test += elbo1_cur * len(batch)

            cost_test = weighted_cost_test / len(test_set)
            costs_test.append(cost_test)

            elbo_test = weighted_elbo_test / len(test_set)
            elbos_test.append(elbo_test)

            elbo1_test = weighted_elbo1_test / len(test_set)
            elbos1_test.append(elbo1_test)

            '''Printing Epoch Time, and Elbo vali'''

            print(str(epoch_counter) + ', ' + str(elbo1_vali) + ', ' + str(time.time() - time_epcoh))
            time_epcoh = time.time()

            '''Plotting During Training'''

            if epoch_counter % epoch_falg == 0:
                sd.save_plot_sample(last_test_gen[:6], "--epoch" + str(epoch_counter).zfill(4),
                                    str(experiment_number).zfill(4) + '-' + str(self.config) + '-' + '-dim_z,' + str(self.dim_z)
                                    + '-beta,' + str(self.beta), path=path_experiment)

                # self.saver.save(self.sess, path_save + '/' + str(experiment_number).zfill(4) + '-epoch--'
                #                 + str(epoch_counter) + '-' + str(self.config) + '-' + '-dim_z,' + str(self.dim_z) +
                #                 '-beta,' + str(self.beta) + '-mult,' + str(self.mult))


                # plt.plot(last_test_batch[1].reshape(np.prod(self.D)))
                # plt.plot(last_test_gen[1].reshape(np.prod(self.D)))
                # plt.savefig(path_experiment + '/' + str(experiment_number) + '-' + str(self.config) + '-' + '-dim_z,' +
                #             str(self.dim_z) + '-beta,' + str(self.beta) + '-mult,' + str(self.mult) + '--epcoh' + str(epoch_counter).zfill(4)+'-compared.jpg')

            # if epoch_counter == 100:
            #     break

            epoch_counter += 1

        '''Plotting, Saving after Training'''

        if not early_termination:
            # Plotting costs curves
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(1, 1, 1)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Cost')
            ax1.set_title('Cost curves')

            ax1.plot(range(len(costs_train)), costs_train, label='training')
            ax1.plot(range(len(costs_vali)), costs_vali, label='validation')
            ax1.plot(range(len(costs_test)), costs_test, label='test')
            ax1.legend(loc="upper right")
            # plt.show()
            fig1.savefig(path_experiment + '/' + str(experiment_number).zfill(4) + "-cost_curves.png", dpi=300)

            # Plotting elbo curves
            # plt.figure()
            # plt.xlabel('Epoch')
            # plt.ylabel('Elbo')
            # plt.title('Elbo curves')
            # plt.plot(range(len(elbos_train)), elbos_train, label='training')
            # plt.plot(range(len(elbos_vali)), elbos_vali, label='validation')
            # plt.plot(range(len(elbos_test)), elbos_test, label='test')
            # plt.legend(loc="upper right")
            # # plt.show()
            # plt.savefig(path_experiment + '/' + str(experiment_number) + "-elbo_curves.png", dpi=300)

            # Plotting elbo1 curves
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1, 1, 1)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Elbo1')
            ax2.set_title('Elbo1 curves')
            ax2.plot(range(len(elbos1_train)), elbos1_train, label='training')
            ax2.plot(range(len(elbos1_vali)), elbos1_vali, label='validation')
            ax2.plot(range(len(elbos1_test)), elbos1_test, label='test')
            ax2.legend(loc="upper right")
            # plt.show()
            fig2.savefig(path_experiment + '/' + str(experiment_number).zfill(4) + "-elbo1_curves.png", dpi=300)

        total_time = time.time() - start_time

        to_store = {'input_type': input_type, 'config': self.config, 'costs_train': costs_train, 'costs_val': costs_vali,
                    'costs_test': costs_test,
                    'best_vali_cost': best_cost_vali, 'best_elbo_vali': best_elbo_vali,
                        'best_elbo1_vali': best_elbo1_vali, 'total_time': total_time}

        with open(path_experiment + '/' + str(experiment_number).zfill(4) + '-' + str(self.config) + '-' + '-dim_z,' + str(self.dim_z)
                    + '-beta,' + str(self.beta) + '-Saved_costs' + ".json",
                  "w") as f:
            json.dump(to_store, f)

        self.saver.save(self.sess, path_save + '/'+str(experiment_number).zfill(4) + '-' + str(self.config) + '-' + '-dim_z,'
                        + str(self.dim_z) + '-beta,' + str(self.beta))

        # Clean up graph and saver object for the next save. Otherwise trash would be saved.
        tf.reset_default_graph()

        return best_elbo1_vali, early_termination



























