"""The code runs, but it doesn't get the same results"""

import tensorflow as tf
from tensorflow.compat.v1 import variable_scope, placeholder, add_to_collection, global_variables, global_variables_initializer, get_collection, train#, layers
import os
from scipy.spatial.distance import pdist, squareform
import numpy as np

print(f"MAGAN is running on TensorFlow {tf.__version__}")

#To run with the older functions, we must include this:
tf.compat.v1.disable_eager_execution()

def nameop(op, name):
    """Give the current op this name, so it can be retrieved in another session."""
    op = tf.identity(op, name=name)
    return op

def tbn(name):
    """Get a tensor of the given name from the graph."""
    return tf.compat.v1.get_default_graph().get_tensor_by_name(name)

def obn(name):
    """Get an object of the given name from the graph."""
    return tf.compat.v1.get_default_graph().get_operation_by_name(name)

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
        activation= "leaky_relu",
        learning_rate=.01, #Used to be 0.0001
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

        self.xb1 = placeholder(tf.float32, shape=[None, self.dim_b1], name='xb1')
        self.xb2 = placeholder(tf.float32, shape=[None, self.dim_b2], name='xb2')

        self.lr = placeholder(tf.float32, shape=[], name='lr')
        self.is_training = placeholder(tf.bool, shape=[], name='is_training')

        self._build()
        self.init_session(limit_gpu_fraction=limit_gpu_fraction, no_gpu=no_gpu)
        self.graph_init(self.sess)

    def init_session(self, limit_gpu_fraction=.4, no_gpu=False):
        """Initialize the session."""
        if no_gpu:
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.compat.v1.Session(config=config)
        elif limit_gpu_fraction:
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.compat.v1.Session(config=config)
        else:
            self.sess = tf.compat.v1.Session()

    def graph_init(self, sess=None):
        """Initialize graph variables."""
        if not sess: sess = self.sess

        self.saver = train.Saver(global_variables(), max_to_keep=1)

        sess.run(global_variables_initializer())

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
        ckpt = train.get_checkpoint_state(restore_folder)
        self.saver = train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
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

        self.xb2_reconstructed = self.G12(self.Gb1)
        self.xb1_reconstructed = self.G21(self.Gb2)
        self.xb1_reconstructed = nameop(self.xb1_reconstructed, 'xb1_reconstructed')
        self.xb2_reconstructed = nameop(self.xb2_reconstructed, 'xb2_reconstructed')

        self.D1 = Discriminator(name='D1')
        self.D2 = Discriminator(name='D2')

        self.D1_probs_z = self.D1(self.xb1)
        self.D1_probs_G = self.D1(self.Gb1)

        self.D2_probs_z = self.D2(self.xb2)
        self.D2_probs_G = self.D2(self.Gb2)

        self._build_loss()

        self._build_optimization()

    def _build_loss(self):
        """Collect both of the losses."""
        self._build_loss_D()
        self._build_loss_G()
        self.loss_D = nameop(self.loss_D, 'loss_D')
        self.loss_G = nameop(self.loss_G, 'loss_G')
        add_to_collection('losses', self.loss_D)
        add_to_collection('losses', self.loss_G)

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
        Gvars = [tv for tv in global_variables() if 'G12' in tv.name or 'G21' in tv.name]
        Dvars = [tv for tv in global_variables() if 'D1' in tv.name or 'D2' in tv.name]

        update_ops = get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        G_update_ops = [op for op in update_ops if 'G12' in op.name or 'G21' in op.name]
        D_update_ops = [op for op in update_ops if 'D1' in op.name or 'D2' in op.name]

        with tf.control_dependencies(G_update_ops):
            optG = train.AdamOptimizer(self.lr, beta1=.5, beta2=.99)
            self.train_op_G = optG.minimize(self.loss_G, var_list=Gvars, name='train_op_G')

        with tf.control_dependencies(D_update_ops):
            optD = train.AdamOptimizer(self.lr, beta1=.5, beta2=.99)
            self.train_op_D = optD.minimize(self.loss_D, var_list=Dvars, name='train_op_D')

    def train_model(self, xb1, xb2):
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
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def get_loss(self, xb1, xb2):
        """Return all of the loss values for the given input."""
        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        ls = [tns for tns in get_collection('losses')]
        losses = self.sess.run(ls, feed_dict=feed)

        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])

        return lstring

class Generator(tf.keras.Model):
    """MAGAN's generator."""

    def __init__(self, output_dim, name='', activation=tf.nn.relu):
        """Initialize the generator."""
        super(Generator, self).__init__(name=name)
        self.output_dim = output_dim
        self.activation = activation

        # Define the layers
        self.dense1 = tf.keras.layers.Dense(200, activation=self.activation, name=f'{name}_h1')
        self.dense2 = tf.keras.layers.Dense(100, activation=self.activation, name=f'{name}_h2')
        self.dense3 = tf.keras.layers.Dense(50, activation=self.activation, name=f'{name}_h3')
        self.out_layer = tf.keras.layers.Dense(self.output_dim, activation=None, name=f'{name}_out')

    def call(self, inputs):
        """Perform the feedforward for the generator."""
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        out = self.out_layer(h3)
        return out
    
class Discriminator(tf.keras.Model):
    """Discriminator model."""

    def __init__(self, name='', activation=tf.nn.relu):
        """Initialize the discriminator."""
        super(Discriminator, self).__init__(name=name)
        self.activation = activation

        # Define the layers
        self.dense1 = tf.keras.layers.Dense(800, activation=self.activation, name=f'{name}_h1')
        self.dense2 = tf.keras.layers.Dense(400, activation=self.activation, name=f'{name}_h2')
        self.dense3 = tf.keras.layers.Dense(200, activation=self.activation, name=f'{name}_h3')
        self.dense4 = tf.keras.layers.Dense(100, activation=self.activation, name=f'{name}_h4')
        self.dense5 = tf.keras.layers.Dense(50, activation=self.activation, name=f'{name}_h5')
        self.out_layer = tf.keras.layers.Dense(1, activation=None, name=f'{name}_out')

    def call(self, inputs):
        """Perform the feedforward for the discriminator."""
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        h4 = self.dense4(h3)
        h5 = self.dense5(h4)
        out = self.out_layer(h5)
        return out


"""Tests Below"""
def get_pure_distance(domain_A, domain_B):
    #Just using a normal distance matrix without Igraph
    x_dists = squareform(pdist(domain_A))
    y_dists = squareform(pdist(domain_B))

    #normalize it
    x_dists = x_dists / np.max(x_dists, axis = None)
    y_dists = y_dists / np.max(y_dists, axis = None)

    return x_dists, y_dists

def get_data(n_batches=2, n_pts_per_cluster=5000): #This only provides two features
    """Return the artificial data."""
    make = lambda x,y,s: np.concatenate([np.random.normal(x,s, (n_pts_per_cluster, 1)), np.random.normal(y,s, (n_pts_per_cluster, 1))], axis=1)
    # batch 1
    xb1 = np.concatenate([make(-1.3, 2.2, .1), make(.1, 1.8, .1), make(.8, 2, .1)], axis=0)
    labels1 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)

    # batch 2
    xb2 = np.concatenate([make(-.9, -2, .1), make(0, -2.3, .1), make(1.5, -1.5, .1)], axis=0)
    labels2 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)

    return xb1, xb2, labels1, labels2

def run_MAGAN(xb1, xb2, labels1, labels2 = "None"): #NOTE: Maybe Magan is expectin xb1 to be a 2 dimensional domain only? 
    """xb1 should be split_a
    sb2 should be split_b
    labels1 should just be the labels"""

    if type(labels2) == type("None"):
        labels2 = labels1
    
    print("Batch 1 shape: {} Batch 2 shape: {}".format(xb1.shape, xb2.shape))

    # Prepare the loaders
    loadb1 = Loader(xb1, labels=labels1, shuffle=True)
    loadb2 = Loader(xb2, labels=labels2, shuffle=True)
    batch_size = np.gcd(len(xb1), 100) #This is changed --- In an attempt to keep the resulting size equivalent to what it began with

    # Build the tf graph
    magan = MAGAN(dim_b1=xb1.shape[1], dim_b2=xb2.shape[1], correspondence_loss=correspondence_loss)

    # Train
    for i in range(1, 2500): #Used to be 100000
        xb1_, labels1_ = loadb1.next_batch(batch_size)
        xb2_, labels2_ = loadb2.next_batch(batch_size)

        magan.train_model(xb1_, xb2_)

        # Evaluate the loss and plot
        if i % 500 == 0:
            xb1_, labels1_ = loadb1.next_batch(len(xb1))
            xb2_, labels2_ = loadb2.next_batch(len(xb1))

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
            axes[0, 0].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[0, 0].scatter(0,0, s=100, c='w'); axes[0, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5])
            axes[0, 1].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[0, 1].scatter(0,0, s=100, c='w'); axes[0, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5])
            axes[1, 0].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[1, 0].scatter(0,0, s=100, c='w'); axes[1, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5])
            axes[1, 1].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[1, 1].scatter(0,0, s=100, c='w'); axes[1, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5])

            for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
                axes[0, 0].scatter(xb1[labels1_ == lab, 0], xb1[labels1_ == lab, 1], s=45, alpha=.5, c='b', marker=marker)
                axes[0, 1].scatter(Gb2[labels1_ == lab, 0], Gb2[labels1_ == lab, 1], s=45, alpha=.5, c='r', marker=marker)
            for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
                axes[1, 0].scatter(xb2[labels2_ == lab, 0], xb2[labels2_ == lab, 1], s=45, alpha=.5, c='r', marker=marker)
                axes[1, 1].scatter(Gb1[labels2_ == lab, 0], Gb1[labels2_ == lab, 1], s=45, alpha=.5, c='b', marker=marker)
            fig.canvas.draw()
            plt.pause(1)

    #Thoughts to understand MAGAN
    """
    xb1 and xb2 are the original data. Gb1 and Gb2 are the generated Data. They are shaped like the data, but all the other methods so 
    far we have been using distance matricies. 

    So we can return Gb1 and Gb2, and then apply our trustee pdist + squareform combo on each of them to get domains. 

    That would leave us something like 
    np.block([[ block_xb1, block_Gb1],
              [ block_Gb2, block_xb2]])

    In the which we could apply our MDS too, for the CE score. 

    It would also return two FOSCTTM scores: 1 for the block_Gb1 and 1 for the block_Gb2. 

    Is that what was intended from this? 
    
    """

    return xb1, xb2, Gb1, Gb2 

"""import test_manifold_algorithms as tma
test = tma.test_manifold_algorithms(csv_file="iris.csv", split = "turn", percent_of_anchors = [0.05], random_state=42, verbose = 2)


magan_model = run_MAGAN(test.split_A, test.split_B, labels1 = test.labels)"""