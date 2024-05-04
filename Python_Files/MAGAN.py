import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import os
import numpy as np
from tensorflow.keras import Model

def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

def correspondence_loss(b1, b2):
    """
    The correspondence loss.

    :param b1: a tensor representing the object in the graph of the current minibatch from domain one
    :param b2: a tensor representing the object in the graph of the current minibatch from domain two
    :returns a scalar tensor of the correspondence loss
    """
    domain1cols = [0]
    domain2cols = [0]
    loss = tf.constant(0.)
    for c1, c2 in zip(domain1cols, domain2cols):
        loss += tf.reduce_mean((b1[:, c1] - b2[:, c2])**2)

    return loss

import numpy as np

class Loader(object):
    """A Loader class for feeding numpy matrices into tensorflow models."""

    def __init__(self, data, labels=None, shuffle=False):
        """Initialize the loader with data and optionally with labels."""
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels] if x is not None]
        self.labels_given = False if labels is None else True

        if shuffle:
            self.r = np.arange(data.shape[0])
            np.random.shuffle(self.r)
            self.data = [x[self.r] for x in self.data]

    def next_batch(self, batch_size=100):
        """Yield the next batch."""
        num_rows = self.data[0].shape[0]

        if self.start + batch_size < num_rows:
            batch = [x[self.start:self.start + batch_size] for x in self.data]
            self.start += batch_size
        else:
            self.epoch += 1
            batch_part1 = [x[self.start:] for x in self.data]
            batch_part2 = [x[:batch_size - (x.shape[0] - self.start)] for x in self.data]
            batch = [np.concatenate([x1, x2], axis=0) for x1, x2 in zip(batch_part1, batch_part2)]

            self.start = batch_size - (num_rows - self.start)

        if not self.labels_given:  # don't return length-1 list
            return batch[0]
        else:  # return list of data and labels
            return batch

    def iter_batches(self, batch_size=100):
        """Iterate over the entire dataset in batches."""
        num_rows = self.data[0].shape[0]

        start = 0
        end = batch_size

        for i in range(num_rows // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size

            if not self.labels_given:
                yield [x[start:end] for x in self.data][0]
            else:
                yield [x[start:end] for x in self.data]

        if batch_size > num_rows:
            if not self.labels_given:
                yield [x for x in self.data][0]
            else:
                yield [x for x in self.data]
        if end != num_rows:
            if not self.labels_given:
                yield [x[end:] for x in self.data][0]
            else:
                yield [x[end:] for x in self.data]


class MAGAN(tf.keras.Model):
    def __init__(self,
                 dim_b1,
                 dim_b2,
                 correspondence_loss,
                 activation=lrelu,
                 learning_rate=.001,
                 limit_gpu_fraction=1.,
                 no_gpu=False,
                 nfilt=64):
        super(MAGAN, self).__init__()
        self.dim_b1 = dim_b1
        self.dim_b2 = dim_b2
        self.correspondence_loss = correspondence_loss
        self.activation = activation
        self.learning_rate = learning_rate
        self.iteration = 0

        self.xb1 = Input(shape=(self.dim_b1,), name='xb1')
        self.xb2 = Input(shape=(self.dim_b2,), name='xb2')
        self.lr = K.variable(self.learning_rate)
        self.is_training = K.variable(True)

        self.G12 = Generator(dim_b2, name='G12')
        self.Gb2 = self.G12(self.xb1)
        self.G21 = Generator(dim_b1, name='G21')
        self.Gb1 = self.G21(self.xb2)

        self.D1 = Discriminator(name='D1')
        self.D2 = Discriminator(name='D2')

        self._build_loss()
        self._build_optimization()

    def _build_loss(self):
        self.D1_probs_z = self.D1(self.xb1)
        self.D1_probs_G = self.D1(self.Gb1)
        self.D2_probs_z = self.D2(self.xb2)
        self.D2_probs_G = self.D2(self.Gb2)

        self.loss_D = (
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_z, labels=tf.ones_like(self.D1_probs_z))) +
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_z, labels=tf.ones_like(self.D2_probs_z))) +
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.zeros_like(self.D1_probs_G))) +
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.zeros_like(self.D2_probs_G)))
        )

        self.loss_G = (
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.ones_like(self.D1_probs_G))) +
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.ones_like(self.D2_probs_G))) +
            tf.reduce_mean(tf.square(self.xb1 - self.G12(self.Gb1))) + tf.reduce_mean(tf.square(self.xb2 - self.G21(self.Gb2))) +
            tf.reduce_mean(self.correspondence_loss(self.xb1, self.Gb2)) +
            tf.reduce_mean(self.correspondence_loss(self.xb2, self.Gb1))
        )

    def _build_optimization(self):
        optG = Adam(self.lr, beta_1=0.5, beta_2=0.99)
        optD = Adam(self.lr, beta_1=0.5, beta_2=0.99)

        @tf.function
        def train_step_G(xb1, xb2):
            with tf.GradientTape() as tape:
                loss_G = self.loss_G(xb1, xb2)
            gradients_G = tape.gradient(loss_G, self.G12.trainable_variables + self.G21.trainable_variables)
            optG.apply_gradients(zip(gradients_G, self.G12.trainable_variables + self.G21.trainable_variables))
            return loss_G

        @tf.function
        def train_step_D(xb1, xb2):
            with tf.GradientTape() as tape:
                loss_D = self.loss_D(xb1, xb2)
            gradients_D = tape.gradient(loss_D, self.D1.trainable_variables + self.D2.trainable_variables)
            optD.apply_gradients(zip(gradients_D, self.D1.trainable_variables + self.D2.trainable_variables))
            return loss_D

        self.train_op_G = train_step_G
        self.train_op_D = train_step_D

    def train(self, xb1, xb2):
        self.iteration += 1
        self.lr.assign(self.learning_rate)
        self.is_training.assign(True)

        feed = {"xb1": xb1, "xb2": xb2}

        self.train_op_G(xb1, xb2)
        self.train_op_D(xb1, xb2)

    def get_layer(self, xb1, xb2, name):
        tensor_name = "{}:0".format(name)
        tensor = self.get_layer(tensor_name)
        feed = {"xb1": xb1, "xb2": xb2}
        layer = tensor.eval(feed['xb1'], feed['xb2'])

        return layer

    def get_loss(self, xb1, xb2):
        feed = {"xb1": xb1, "xb2": xb2}

        loss_D, loss_G = self.sess.run([self.loss_D, self.loss_G], feed['xb1'], feed['xb2'])

        lstring = ' '.join(['{:.3f}'.format(loss) for loss in [loss_D, loss_G]])

        return lstring

class Generator(tf.keras.Model):
    def __init__(self, output_dim, name='', activation=tf.nn.relu):
        super(Generator, self).__init__(name=name)
        self.output_dim = output_dim
        self.activation = activation
        self.dense1 = Dense(200, activation=activation, name='h1')
        self.dense2 = Dense(100, activation=activation, name='h2')
        self.dense3 = Dense(50, activation=activation, name='h3')
        self.dense4 = Dense(output_dim, activation=None, name='out')

    def call(self, x, training=False):
        h1 = self.dense1(x)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        out = self.dense4(h3)
        return out

class Discriminator(tf.keras.Model):
    def __init__(self, name='', activation=tf.nn.relu):
        super(Discriminator, self).__init__(name=name)
        self.activation = activation
        self.dense1 = Dense(800, activation=activation, name='h1')
        self.dense2 = Dense(400, activation=activation, name='h2')
        self.dense3 = Dense(200, activation=activation, name='h3')
        self.dense4 = Dense(100, activation=activation, name='h4')
        self.dense5 = Dense(50, activation=activation, name='h5')
        self.dense6 = Dense(1, activation=None, name='out')

    def call(self, x, training=False):
        h1 = self.dense1(x)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        h4 = self.dense4(h3)
        h5 = self.dense5(h4)
        out = self.dense6(h5)
        return out































