import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_probability as tfp
import functools
import time

import matplotlib.pyplot as plt

class TimeVAE:
    def make_encoder(self, data, code_size):
        x = tf.layers.flatten(data)

        # x = tf.layers.dense(x, 70, tf.nn.relu)
        x = tf.layers.dense(x, 50, tf.nn.relu)
        x = tf.layers.dense(x, 20, tf.nn.relu)
        # x = tf.layers.dense(x, 20, tf.nn.relu)
        x = tf.layers.dense(x, 10, tf.nn.relu)
        # x = tf.layers.dense(x, 10, tf.nn.relu)

        loc = tf.layers.dense(x, code_size)
        scale = tf.layers.dense(x, code_size, tf.nn.softplus)
        # return tfd.MultivariateNormalDiag(loc, scale)
        return tfp.distributions.MultivariateNormalDiag(loc, scale)

    def make_prior(self, code_size):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        # return tfd.MultivariateNormalDiag(loc, scale)
        return tfp.distributions.MultivariateNormalDiag(loc, scale)


    def make_decoder(self, code, data_shape):
        x = code

        x = tf.layers.dense(x, 10, tf.nn.relu)
        # x = tf.layers.dense(x, 10, tf.nn.relu)
        x = tf.layers.dense(x, 20, tf.nn.relu)
        # x = tf.layers.dense(x, 20, tf.nn.relu)
        x = tf.layers.dense(x, 50, tf.nn.relu)
        # x = tf.layers.dense(x, 70, tf.nn.relu)

        logit = tf.layers.dense(x, np.prod(data_shape))
        logit = tf.reshape(logit, [-1] + data_shape)
        # return tfd.Independent(tfd.Bernoulli(logit), 2)
        return tfp.distributions.Independent(tfp.distributions.Bernoulli(logit))

    def __init__(self, D, dim_z, beta):

        # self.sess = sess
        # self.saver = saver

        self.beta = beta
        self.dim_z = dim_z
        self.D = D

        self.encoder = tf.make_template('encoder', self.make_encoder)
        self.decoder = tf.make_template('decoder', self.make_decoder)

        # self.data = tf.placeholder(tf.float32, shape=(None, 28, 28))
        # self.data = tf.placeholder(tf.float32, shape=(None, 10, 10))
        self.data = tf.placeholder(tf.float32, shape=[None] + D)


        self.prior = self.make_prior(code_size=dim_z)
        self.posterior = self.encoder(self.data, code_size=dim_z)
        self.code = self.posterior.sample()

        # self.samples = self.decoder(self.prior.sample(10), [28, 28]).mean()
        # self.samples = self.decoder(self.prior.sample(10), [10, 10]).mean()
        self.samples = self.decoder(self.prior.sample(10), D).mean()


        # self.likelihood = self.decoder(self.code, [28, 28]).log_prob(self.data)
        # self.likelihood = self.decoder(self.code, [10, 10]).log_prob(self.data)
        self.likelihood = self.decoder(self.code, D).log_prob(self.data)

        self.divergence = tfp.distributions.kl_divergence(self.posterior, self.prior)
        self.elbo = tf.reduce_mean(self.likelihood - self.beta*self.divergence)

        self.optimize = tf.train.AdamOptimizer(0.001).minimize(-self.elbo)

        self.tf_code = tf.placeholder(dtype=tf.float32, shape=[None, dim_z])
        # self.test_decoded = self.decoder(self.tf_code, [28, 28]).mean()
        # self.test_decoded = self.decoder(self.tf_code, [10, 10]).mean()
        self.test_decoded = self.decoder(self.tf_code, D).mean()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)


    def fit(self, X, test, epochs=500, batch_sz=100):
        code = []
        n_batches = len(X) // batch_sz
        current_time = time.time()
        test = test.reshape([len(test)] + self.D)
        X = X.reshape([len(X)] + self.D)
        for i in range(epochs):
            test_elbo, test_codes, test_samples, opt = self.sess.run(
                [self.elbo, self.code, self.samples, self.optimize], {self.data: test})
            print('Epoch', i, 'elbo', test_elbo)
            print(time.time() - current_time)
            current_time = time.time()
            # plt.imshow(test_samples[1])

            plt.plot(test[1].reshape(np.prod(self.D)))
            plt.plot(test_samples[1].reshape(np.prod(self.D)))
            # plt.savefig("plots/first/compared-" + str(i).zfill(4) + ".jpg")
            plt.show()


            print("DVAE- epoch:", i)
            np.random.shuffle(X)

            for j in range(n_batches):
                batch = X[j * batch_sz:(j + 1) * batch_sz]

                _, code = self.sess.run(fetches=(self.optimize, self.code), feed_dict={self.data: batch})

        self.saver.save(self.sess, 'Saved-sess/dvae-1-70-50-2_20-2_10-3')
        print('dvae-1-70-50-2_20-2_10-3' + 'is saved.')
        return code

























