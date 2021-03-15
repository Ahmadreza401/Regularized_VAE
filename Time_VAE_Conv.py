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
        x = tf.layers.conv2d(inputs=tf.reshape(data, shape=[-1, 10, 10, 1]), filters=2,
                             kernel_size=[2, 2], strides=(2, 2),
                             activation=tf.nn.relu, padding="same")  # 10*10*1 -> 5*5*2: 100 -> 50
        x = tf.layers.conv2d(inputs=x, filters=2, kernel_size=[2, 2],
                             strides=(1, 1), activation=tf.nn.relu, padding="same")  # 50 -> 50
        x = tf.layers.conv2d(inputs=x, filters=20, kernel_size=[5, 5],      # 5*5*2 -> 1*1*20: 50 ->20
                             activation=tf.nn.relu, padding="valid")
        x = tf.layers.dense(inputs=x, units=2*3)   # 20 -> 6

        loc = x[..., :3]
        scale = tf.nn.softplus(x[..., 3:]) + 1e-6
        # return tfd.MultivariateNormalDiag(loc, scale)
        return tfp.distributions.MultivariateNormalDiag(loc, scale)

    def make_prior(self, code_size):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        # return tfd.MultivariateNormalDiag(loc, scale)
        return tfp.distributions.MultivariateNormalDiag(loc, scale)


    def make_decoder(self, code, data_shape):
        x = tf.reshape(code, shape=[-1, 1, 1, 3])
        x = tf.layers.dense(inputs=x, units=20)  # 3->20
        x = tf.layers.conv2d_transpose(inputs=x, filters=2, kernel_size=[5, 5],
                                       activation=tf.nn.relu, padding="valid")  # 1, 1, 20 -> 5, 5, 2: 20 -> 50
        x = tf.layers.conv2d_transpose(inputs=x, filters=2, kernel_size=[2, 2], strides=(1, 1),
                                       activation=tf.nn.relu, padding="same")   # 50 -> 50
        x = tf.layers.conv2d_transpose(inputs=x, filters=1, kernel_size=[2, 2], strides=(2, 2),
                                       activation=tf.nn.relu, padding="same")   # 50 -> 100

        logit = tf.reshape(x, [-1] + data_shape)
        # return tfd.Independent(tfd.Bernoulli(logit), 2)
        return tfp.distributions.Independent(tfp.distributions.Bernoulli(logit))

    def __init__(self, D, dim_z, beta, sess=None, saver=None):

        # self.sess = sess
        # self.saver = saver

        self.beta = beta
        self.dim_z = dim_z
        self.D = D

        self.encoder = tf.make_template('encoder', self.make_encoder)
        self.decoder = tf.make_template('decoder', self.make_decoder)

        self.data = tf.placeholder(tf.float32, shape=[None] + D)
        # self.data = tf.placeholder(tf.float32, shape=(None, 10, 10))


        self.prior = self.make_prior(code_size=dim_z)
        self.posterior = self.encoder(self.data, code_size=dim_z)
        self.code = self.posterior.sample()

        self.samples = self.decoder(self.prior.sample(10), D).mean()
        # self.samples = self.decoder(self.prior.sample(10), [10, 10]).mean()

        self.likelihood = self.decoder(self.code, D).log_prob(self.data)
        # self.likelihood = self.decoder(self.code, [10, 10]).log_prob(self.data)


        self.divergence = tfp.distributions.kl_divergence(self.posterior, self.prior)
        self.elbo = tf.reduce_mean(self.likelihood - self.beta*self.divergence)

        self.optimize = tf.train.AdamOptimizer(0.001).minimize(-self.elbo)

        self.tf_code = tf.placeholder(dtype=tf.float32, shape=[None, dim_z])

        self.test_decoded = self.decoder(self.tf_code, D).mean()
        # self.test_decoded = self.decoder(self.tf_code, [10, 10]).mean()


        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)


    def fit(self, X, test, epochs=100, batch_sz=100):
        code = []
        n_batches = len(X) // batch_sz
        current_time = time.time()
        test = test.reshape([len(test)] + self.D)
        X = X.reshape([len(X)] + self.D)
        for i in range(epochs):
            test_elbo, test_codes, test_samples = self.sess.run(
                [self.elbo, self.code, self.samples], {self.data: test})
            print('Epoch', i, 'elbo', test_elbo)
            print(time.time() - current_time)
            current_time = time.time()
            # plt.imshow(test_samples[1])

            plt.plot(test[1].reshape(100))
            plt.plot(test_samples[1].reshape([100]))
            # plt.savefig("plots/first/compared-" + str(i).zfill(4) + ".jpg")
            plt.show()


            print("epoch:", i)
            np.random.shuffle(X)

            for j in range(n_batches):
                batch = X[j * batch_sz:(j + 1) * batch_sz]

                _, code = self.sess.run(fetches=(self.optimize, self.code), feed_dict={self.data: batch})

        self.saver.save(self.sess, 'C:/Saved-sess/Conv/cvae-[50, 20]--dim_z,3-')
        print('')
        return code

























