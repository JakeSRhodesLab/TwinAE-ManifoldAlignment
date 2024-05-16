"""NOTE:
Will require creating a new environment to run and older version of tensorflow. Can be accomplished through the following commands

conda create -n tf115 python=3.7  

conda activate tf115   

conda install tensorflow==1.15.0   

python Python_Files/MAGAN.py

conda deactivate

"""


import tensorflow as tf
import os
import numpy as np

print(f"MAGAN is running on TensorFlow {tf.__version__}")

def lrelu(x, leak=0.2, name="lrelu"):
    """A leaky Rectified Linear Unit."""
    return tf.maximum(x, leak * x)

def nameop(op, name):
    """Give the current op this name, so it can be retrieved in another session."""
    op = tf.identity(op, name=name)
    return op

def tbn(name):
    """Get a tensor of the given name from the graph."""
    return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):
    """Get an object of the given name from the graph."""
    return tf.get_default_graph().get_operation_by_name(name)

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

class Loader(object):
    """A Loader class for feeding numpy matrices into tensorflow models."""

    def __init__(self, data, labels=None, shuffle=False):
        """Initialize the loader with data and optionally with labels."""
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels] if x is not None]
        self.labels_given = False if labels is None else True

        if shuffle:
            self.r = list(range(data.shape[0]))
            np.random.shuffle(self.r)
            self.data = [x[self.r] for x in self.data]

    def next_batch(self, batch_size=100):
        """Yield just the next batch."""
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

class MAGAN(object):
    """The MAGAN model."""

    def __init__(self,
        dim_b1,
        dim_b2,
        correspondence_loss,
        activation=lrelu,
        learning_rate=.001,
        restore_folder='',
        limit_gpu_fraction=1.,
        no_gpu=False,
        nfilt=64):
        """Initialize the model."""
        self.dim_b1 = dim_b1
        self.dim_b2 = dim_b2
        self.correspondence_loss = correspondence_loss
        self.activation = activation
        self.learning_rate = learning_rate
        self.iteration = 0

        if restore_folder:
            self._restore(restore_folder)
            return

        self.xb1 = tf.placeholder(tf.float32, shape=[None, self.dim_b1], name='xb1')
        self.xb2 = tf.placeholder(tf.float32, shape=[None, self.dim_b2], name='xb2')

        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        self._build()
        self.init_session(limit_gpu_fraction=limit_gpu_fraction, no_gpu=no_gpu)
        self.graph_init(self.sess)

    def init_session(self, limit_gpu_fraction=.4, no_gpu=False):
        """Initialize the session."""
        if no_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        elif limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

    def graph_init(self, sess=None):
        """Initialize graph variables."""
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())

        return self.saver

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        """Save the model."""
        if not iteration: iteration = self.iteration
        if not saver: saver = self.saver
        if not sess: sess = self.sess
        if not folder: folder = self.save_folder

        savefile = os.path.join(folder, 'MAGAN')
        saver.save(sess, savefile, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def _restore(self, restore_folder):
        """Restore the model from a saved checkpoint."""
        tf.reset_default_graph()
        self.init_session()
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored from {}".format(restore_folder))

    def _build(self):
        """Construct the DiscoGAN operations."""
        self.G12 = Generator(self.dim_b2, name='G12')
        self.Gb2 = self.G12(self.xb1)
        self.Gb2 = nameop(self.Gb2, 'Gb2')

        self.G21 = Generator(self.dim_b1, name='G21')
        self.Gb1 = self.G21(self.xb2)
        self.Gb1 = nameop(self.Gb1, 'Gb1')

        self.xb2_reconstructed = self.G12(self.Gb1, reuse=True)
        self.xb1_reconstructed = self.G21(self.Gb2, reuse=True)
        self.xb1_reconstructed = nameop(self.xb1_reconstructed, 'xb1_reconstructed')
        self.xb2_reconstructed = nameop(self.xb2_reconstructed, 'xb2_reconstructed')

        self.D1 = Discriminator(name='D1')
        self.D2 = Discriminator(name='D2')

        self.D1_probs_z = self.D1(self.xb1)
        self.D1_probs_G = self.D1(self.Gb1, reuse=True)

        self.D2_probs_z = self.D2(self.xb2)
        self.D2_probs_G = self.D2(self.Gb2, reuse=True)

        self._build_loss()

        self._build_optimization()

    def _build_loss(self):
        """Collect both of the losses."""
        self._build_loss_D()
        self._build_loss_G()
        self.loss_D = nameop(self.loss_D, 'loss_D')
        self.loss_G = nameop(self.loss_G, 'loss_G')
        tf.add_to_collection('losses', self.loss_D)
        tf.add_to_collection('losses', self.loss_G)

    def _build_loss_D(self):
        """Discriminator loss."""
        losses = []
        # the true examples
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_z, labels=tf.ones_like(self.D1_probs_z))))
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_z, labels=tf.ones_like(self.D2_probs_z))))
        # the generated examples
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.zeros_like(self.D1_probs_G))))
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.zeros_like(self.D2_probs_G))))
        self.loss_D = tf.reduce_mean(losses)

    def _build_loss_G(self):
        """Generator loss."""
        losses = []
        # fool the discriminator losses
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.ones_like(self.D1_probs_G))))
        losses.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.ones_like(self.D2_probs_G))))
        # reconstruction losses
        losses.append(tf.reduce_mean((self.xb1 - self.xb1_reconstructed)**2))
        losses.append(tf.reduce_mean((self.xb2 - self.xb2_reconstructed)**2))
        # correspondences losses
        losses.append(1 * tf.reduce_mean(self.correspondence_loss(self.xb1, self.Gb2)))
        losses.append(1 * tf.reduce_mean(self.correspondence_loss(self.xb2, self.Gb1)))

        self.loss_G = tf.reduce_mean(losses)

    def _build_optimization(self):
        """Build optimization components."""
        Gvars = [tv for tv in tf.global_variables() if 'G12' in tv.name or 'G21' in tv.name]
        Dvars = [tv for tv in tf.global_variables() if 'D1' in tv.name or 'D2' in tv.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        G_update_ops = [op for op in update_ops if 'G12' in op.name or 'G21' in op.name]
        D_update_ops = [op for op in update_ops if 'D1' in op.name or 'D2' in op.name]

        with tf.control_dependencies(G_update_ops):
            optG = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.99)
            self.train_op_G = optG.minimize(self.loss_G, var_list=Gvars, name='train_op_G')

        with tf.control_dependencies(D_update_ops):
            optD = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.99)
            self.train_op_D = optD.minimize(self.loss_D, var_list=Dvars, name='train_op_D')

    def train(self, xb1, xb2):
        """Take a training step with batches from each domain."""
        self.iteration += 1

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('lr:0'): self.learning_rate,
                tbn('is_training:0'): True}

        _ = self.sess.run([obn('train_op_G')], feed_dict=feed)
        _ = self.sess.run([obn('train_op_D')], feed_dict=feed)

    def get_layer(self, xb1, xb2, name):
        """Get a layer of the network by name for the entire datasets given in xb1 and xb2."""
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        layer = self.sess.run(tensor, feed_dict=feed)

        return layer

    def get_loss_names(self):
        """Return a string for the names of the loss values."""
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def get_loss(self, xb1, xb2):
        """Return all of the loss values for the given input."""
        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        ls = [tns for tns in tf.get_collection('losses')]
        losses = self.sess.run(ls, feed_dict=feed)

        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])

        return lstring

class Generator(object):
    """MAGAN's generator."""

    def __init__(self,
        output_dim,
        name='',
        activation=tf.nn.relu):
        """"Initialize the generator."""
        self.output_dim = output_dim
        self.activation = activation
        self.name = name

    def __call__(self, x, reuse=False):
        """Perform the feedforward for the generator."""
        with tf.variable_scope(self.name):
            h1 = tf.layers.dense(x, 200, activation=self.activation, reuse=reuse, name='h1')
            h2 = tf.layers.dense(h1, 100, activation=self.activation, reuse=reuse, name='h2')
            h3 = tf.layers.dense(h2, 50, activation=self.activation, reuse=reuse, name='h3')

            out = tf.layers.dense(h3, self.output_dim, activation=None, reuse=reuse, name='out')

        return out

class Discriminator(object):
    """MAGAN's discriminator."""

    def __init__(self,
        name='',
        activation=tf.nn.relu):
        """Initialize the discriminator."""
        self.activation = activation
        self.name = name

    def __call__(self, x, reuse=False):
        """Perform the feedforward for the discriminator."""
        with tf.variable_scope(self.name):
            h1 = tf.layers.dense(x, 800, activation=self.activation, reuse=reuse, name='h1')
            h2 = tf.layers.dense(h1, 400, activation=self.activation, reuse=reuse, name='h2')
            h3 = tf.layers.dense(h2, 200, activation=self.activation, reuse=reuse, name='h3')
            h4 = tf.layers.dense(h3, 100, activation=self.activation, reuse=reuse, name='h4')
            h5 = tf.layers.dense(h4, 50, activation=self.activation, reuse=reuse, name='h5')

            out = tf.layers.dense(h5, 1, activation=None, reuse=reuse, name='out')

        return out


"""Tests Below"""

def get_data(n_batches=2, n_pts_per_cluster=5000):
    """Return the artificial data."""
    make = lambda x,y,s: np.concatenate([np.random.normal(x,s, (n_pts_per_cluster, 1)), np.random.normal(y,s, (n_pts_per_cluster, 1))], axis=1)
    # batch 1
    xb1 = np.concatenate([make(-1.3, 2.2, .1), make(.1, 1.8, .1), make(.8, 2, .1)], axis=0)
    labels1 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)

    # batch 2
    xb2 = np.concatenate([make(-.9, -2, .1), make(0, -2.3, .1), make(1.5, -1.5, .1)], axis=0)
    labels2 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)

    return xb1, xb2, labels1, labels2

# Load the data
xb1, xb2, labels1, labels2 = get_data()


print("Batch 1 shape: {} Batch 2 shape: {}".format(xb1.shape, xb2.shape))

# Prepare the loaders
loadb1 = Loader(xb1, labels=labels1, shuffle=True)
loadb2 = Loader(xb2, labels=labels2, shuffle=True)
batch_size = 100

# Build the tf graph
magan = MAGAN(dim_b1=xb1.shape[1], dim_b2=xb2.shape[1], correspondence_loss=correspondence_loss)

# Train
for i in range(1, 100000):
    xb1_, labels1_ = loadb1.next_batch(batch_size)
    xb2_, labels2_ = loadb2.next_batch(batch_size)

    magan.train(xb1_, xb2_)

    # Evaluate the loss and plot
    if i % 500 == 0:
        xb1_, labels1_ = loadb1.next_batch(10 * batch_size)
        xb2_, labels2_ = loadb2.next_batch(10 * batch_size)

        lstring = magan.get_loss(xb1_, xb2_)
        print("{} {}".format(magan.get_loss_names(), lstring))


        xb1 = magan.get_layer(xb1_, xb2_, 'xb1')
        xb2 = magan.get_layer(xb1_, xb2_, 'xb2')
        Gb1 = magan.get_layer(xb1_, xb2_, 'Gb1')
        Gb2 = magan.get_layer(xb1_, xb2_, 'Gb2')

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        axes[0, 0].set_title('Original')
        axes[0, 1].set_title('Generated')
        axes[0, 0].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[0, 0].scatter(0,0, s=100, c='w'); axes[0, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[0, 1].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[0, 1].scatter(0,0, s=100, c='w'); axes[0, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 0].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[1, 0].scatter(0,0, s=100, c='w'); axes[1, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 1].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[1, 1].scatter(0,0, s=100, c='w'); axes[1, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);

        for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
            axes[0, 0].scatter(xb1[labels1_ == lab, 0], xb1[labels1_ == lab, 1], s=45, alpha=.5, c='b', marker=marker)
            axes[0, 1].scatter(Gb2[labels1_ == lab, 0], Gb2[labels1_ == lab, 1], s=45, alpha=.5, c='r', marker=marker)
        for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
            axes[1, 0].scatter(xb2[labels2_ == lab, 0], xb2[labels2_ == lab, 1], s=45, alpha=.5, c='r', marker=marker)
            axes[1, 1].scatter(Gb1[labels2_ == lab, 0], Gb1[labels2_ == lab, 1], s=45, alpha=.5, c='b', marker=marker)
        fig.canvas.draw()
        plt.pause(1)
