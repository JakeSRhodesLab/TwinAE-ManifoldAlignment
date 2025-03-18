import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import time
import matplotlib
from sklearn.metrics import mean_squared_error

import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42 

# Hyperparameters defaults
BATCH_SIZE = 128
LR = .0001
WEIGHT_DECAY = 0
EPOCHS = 200
HIDDEN_DIMS = (800, 400, 200)  # Default fully-connected dimensions
CONV_DIMS = [32, 64]  # Default conv channels
CONV_FC_DIMS = [400, 200]  # Default fully-connected dimensions after convs
FIT_DEFAULT = .85  # Default train/test split ratio
DEFAULT_PATH = os.path.join(os.getcwd(), 'data')


class FromNumpyDataset(Dataset):
    """Torch Dataset Wrapper for x ndarray with no target."""

    def __init__(self, x):
        """Create torch wraper dataset form simple ndarray.

        Args:
            x (ndarray): Input variables.
        """
        self._data = torch.from_numpy(x).float()

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.

        Specify indices to return a subset of the dataset, otherwise return whole dataset.

        Args:
            idx(int, optional): Specify index or indices to return.

        Returns:
            ndarray: Return flattened dataset as a ndarray.

        """
        n = len(self)

        data = self._data.numpy().reshape((n, -1))

        if idx is None:
            return data
        else:
            return data[idx]

class BaseDataset(Dataset):
    """Template class for all datasets in the project.

    All datasets should subclass BaseDataset, which contains built-in splitting utilities."""

    def __init__(self, x, y, split, split_ratio, random_state, labels=None):
        """Init.

        Set the split parameter to 'train' or 'test' for the object to hold the desired split. split='none' will keep
        the entire dataset in the attributes.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
            split(str): Name of split.
            split_ratio(float): Ratio to use for train split. Test split ratio is 1 - split_ratio.
            random_state(int): To set random_state values for reproducibility.
            labels(ndarray, optional): Specify labels for stratified splits.
        """
        if split not in ('train', 'test', 'none'):
            raise ValueError('split argument should be "train", "test" or "none"')

        # Get train or test split
        x, y = self.get_split(x, y, split, split_ratio, random_state, labels)

        self.data = x.float()
        self.targets = y.float()  # One target variable. Used mainly for coloring.
        self.latents = None  # Arbitrary number of continuous ground truth variables. Used for computing metrics.

        # Arbitrary number of label ground truth variables. Used for computing metrics.
        # Should range from 0 to no_of_classes -1
        self.labels = None
        self.is_radial = []  # Indices of latent variable requiring polar conversion when probing (e.g. Teapot, RotatedDigits)
        self.partition = True  # If labels should be used to partition the data before regressing latent factors. See score.EmbeddingProber.

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index

    def __len__(self):
        return len(self.data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.

        Specify indices to return a subset of the dataset, otherwise return whole dataset.

        Args:
            idx(int, optional): Specify index or indices to return.

        Returns:
            ndarray: Return flattened dataset as a ndarray.

        """
        n = len(self)

        data = self.data.numpy().reshape((n, -1))

        if idx is None:
            return data, self.targets.numpy()
        else:
            return data[idx], self.targets[idx].numpy()

    def get_split(self, x, y, split, split_ratio, random_state, labels=None):
        """Split dataset.

        Args:
            x(ndarray): Input features.
            y(ndarray): Targets.
            split(str): Name of split.
            split_ratio(float): Ratio to use for train split. Test split ratio is 1 - split_ratio.
            random_state(int): To set random_state values for reproducibility.
            labels(ndarray, optional): Specify labels for stratified splits.

        Returns:
            (tuple): tuple containing :
                    x(ndarray): Input variables in requested split.
                    y(ndarray): Target variable in requested split.
        """
        if split == 'none':
            return torch.from_numpy(x), torch.from_numpy(y)

        n = x.shape[0]
        train_idx, test_idx = train_test_split(np.arange(n),
                                               train_size=split_ratio,
                                               random_state=random_state,
                                               stratify=labels)

        if split == 'train':
            return torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])
        else:
            return torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx])

    def get_latents(self):
        """Latent variable getter.

        Returns:
            latents(ndarray): Latent variables for each sample.
        """
        return self.latents

    def random_subset(self, n, random_state):
        """Random subset self and return corresponding dataset object.

        Args:
            n(int): Number of samples to subset.
            random_state(int): Seed for reproducibility

        Returns:
            Subset(TorchDataset) : Random subset.

        """

        np.random.seed(random_state)
        sample_mask = np.random.choice(len(self), n, replace=False)

        next_latents = self.latents[sample_mask] if self.latents is not None else None
        next_labels = self.labels[sample_mask] if self.labels is not None else None

        return NoSplitBaseDataset(self.data[sample_mask], self.targets[sample_mask], next_latents, next_labels)

    def validation_split(self, ratio=.15 / FIT_DEFAULT, random_state=42):
        """Randomly subsample validation split in self.

        Return both train split and validation split as two different BaseDataset objects.

        Args:
            ratio(float): Ratio of train split to allocate to validation split. Default option is to sample 15 % of
            full dataset, by adjusting with the initial train/test ratio.
            random_state(int): Seed for sampling.

        Returns:
            (tuple) tuple containing:
                x_train(BaseDataset): Train set.
                x_val(BaseDataset): Val set.

        """

        np.random.seed(random_state)
        sample_mask = np.random.choice(len(self), int(ratio * len(self)), replace=False)
        val_mask = np.full(len(self), False, dtype=bool)
        val_mask[sample_mask] = True
        train_mask = np.logical_not(val_mask)
        next_latents_train = self.latents[train_mask] if self.latents is not None else None
        next_latents_val = self.latents[val_mask] if self.latents is not None else None
        next_labels_train = self.labels[train_mask] if self.labels is not None else None
        next_labels_val = self.labels[val_mask] if self.labels is not None else None

        x_train = NoSplitBaseDataset(self.data[train_mask], self.targets[train_mask],
                                     next_latents_train, next_labels_train)
        x_val = NoSplitBaseDataset(self.data[val_mask], self.targets[val_mask],
                                   next_latents_val, next_labels_val)

        return x_train, x_val

class NoSplitBaseDataset(BaseDataset):
    """BaseDataset class when splitting is not required and x and y are already torch tensors."""

    def __init__(self, x, y, latents, labels):
        """Init.

        Args:
            x(ndarray): Input variables.
            y(ndarray): Target variable. Used for coloring.
            latents(ndarray): Other continuous target variable. Used for metrics.
            labels(ndarray): Other label target variable. Used for metrics.
        """
        self.data = x.float()
        self.targets = y.float()
        self.latents = latents
        self.labels = labels

"""Parent class for all project models."""
class BaseModel:
    """All models should subclass BaseModel."""
    def __init__(self):
        """Init."""
        self.comet_exp = None

    def fit(self, x):

        raise NotImplementedError()

    def fit_transform(self, x):
        """Fit model and transform data.

        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of X.

        Args:
            X(ndarray or Torch Tensor): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x.

        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        """Transform data.

        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of x.

        Args:
             X(ndarray or Torch Tensor):
        Returns:
            ndarray: Embedding of X.

        """
        raise NotImplementedError()

    def inverse_transform(self, x):
        """Take coordinates in the embedding space and invert them to the data space.

        Args:
            x(ndarray): Points in the embedded space with samples on the first axis.
        Returns:
            ndarray: Inverse (reconstruction) of x.

        """
        raise NotImplementedError()

    def fit_plot(self, x_train, y_train = None, x_test=None, y_test = None, cmap='jet', s=15, title=None):
        """Fit x_train and show a 2D scatter plot of x_train (and possibly x_test).

        If x_test is provided, x_train points will be smaller and grayscale and x_test points will be colored.

        Args:
            x_train(ndarray): Data to fit and plot.
            x_test(ndarray): Data to plot. Set to None to only plot x_train.
            y_train(ndarray): Target variable for coloring.
            y_test(ndarray): Target variable for coloring.
            cmap(str): Matplotlib colormap.
            s(float): Scatter plot marker size.
            title(str): Figure title. Set to None for no title.

        """
        self.plot(x_train, y_train, x_test, y_test, cmap, s, title, fit=True)

    def plot(self, x_train, y_train, x_test=None, y_test = None, cmap='jet', s=15, title=None, fit=False, figsize = (10, 7)):
        """Plot x_train (and possibly x_test) and show a 2D scatter plot of x_train (and possibly x_test).

        If x_test is provided, x_train points will be smaller and grayscale and x_test points will be colored.
        Will log figure to comet if Experiment object is provided. Otherwise, plt.show() is called.

        Args:
            x_train(ndarray): Data to fit and plot.
            x_test(ndarray): Data to plot. Set to None to only plot x_train.
            y_train(ndarray): Target variable for coloring.
            y_test(ndarray): Target variable for coloring.
            cmap(str): Matplotlib colormap.
            s(float): Scatter plot marker size.
            title(str): Figure title. Set to None for no title.

        """
        if self.comet_exp is not None:
            # If comet_exp is set, use different backend to avoid display errors on clusters
            matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!

        import matplotlib.pyplot as plt

        if not fit:
            z_train = self.transform(x_train)
        else:
            z_train = self.fit_transform(x_train)

        if z_train.shape[1] != 2:
            raise Exception('Can only plot 2D embeddings.')

        plt.figure(figsize=figsize)

        if title is not None:
            plt.title(title, fontsize=12)
            plt.xticks([])
            plt.yticks([])

        if x_test is None:
            plt.scatter(*z_train.T, c=y_train, cmap=cmap, s=s)
        else:
            # Train data is grayscale and Test data is colored
            z_test = self.transform(x_test)
            if y_train is None:
                plt.scatter(*z_train.T, s=s / 5, alpha=.7, edgecolors='black', linewidths=0.125)
            else:
                plt.scatter(*z_train.T, c=y_train, s=s / 5, alpha=.7, edgecolors='black', linewidths=0.125)
            if y_test is None:
                plt.scatter(*z_test.T, cmap=cmap, s=s, edgecolors='black', linewidths=0.3)
            else:
                plt.scatter(*z_test.T, c=y_test, cmap=cmap, s=s, edgecolors='black', linewidths=0.3)

        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure=plt, figure_name=title)
            plt.clf()
        else:
            plt.show()

    def reconstruct(self, x):
        """Transform and inverse x.

        Args:
            x(ndarray): Data to transform and reconstruct.

        Returns:
            ndarray: Reconstructions of x.

        """
        return self.inverse_transform(self.transform(x))

    def score(self, x):
        """Compute embedding of x, MSE on x and performance time of transform and inverse transform on x.

        Args:
            x(BaseDataset): Dataset to score.

        Returns:
            (tuple) tuple containing:
                z(ndarray): Data embedding.
                metrics(dict[float]):
                    MSE(float): Reconstruction MSE error.
                    rec_time(float): Reconstruction time in seconds.
                    transform_time(float): Transform time in seconds.
        """
        n = len(x)

        start = time.time()
        z = self.transform(x)
        stop = time.time()

        transform_time = stop - start

        start = time.time()
        x_hat = self.inverse_transform(z)
        stop = time.time()

        rec_time = stop - start

        x, _ = x.numpy()
        MSE = mean_squared_error(x.reshape((n, -1)), x_hat.reshape((n, -1)))

        return z, {
            'MSE': MSE,
            'transform_time': transform_time,
            'rec_time': rec_time,
        }

    def view_img_rec(self, x, n=8, random_state=42, title=None, choice='random'):
        """View n original images and their reconstructions.

        Only call this method on images dataset. x is expected to be 4D.
        Will show figure or log it to Comet if self.comet_exp was set.

        Args:
            x(ndarray): Dataset to sample from.
            n(int): Number of images to sample.
            random_state(int): Seed for sampling.
            title(str): Figure title.
            choice(str): 'random' for n random images in dataset. 'best' for images with best reconstructions. 'worst'
            for images with worst reconstructions.

        """
        if self.comet_exp is not None:
            # If comet_exp is set, use different backend to avoid display errors on clusters
            matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
        import matplotlib.pyplot as plt

        np.random.seed(random_state)

        x_hat = self.reconstruct(x)

        if choice == 'random':
            sample_mask = np.random.choice(x.shape[0], size=n, replace=False)
        else:
            n_sample = x.shape[0]
            mse = mean_squared_error(x.reshape((n_sample, -1)).T, x_hat.reshape((n_sample, -1)).T, multioutput='raw_values')
            mse_rank = np.argsort(mse)
            if choice == 'worst':
                sample_mask = mse_rank[-n:]
            elif choice == 'best':
                sample_mask = mse_rank[:n]
            else:
                raise Exception('Choice name should be random, best or worst.')

        x_hat = x_hat[sample_mask]
        x = x[sample_mask].reshape(x_hat.shape)

        if x_hat.shape[1] == 1:
            grayscale = True
            x_hat = np.squeeze(x_hat)
            x = np.squeeze(x)
        elif x_hat.shape[1] == 3:
            grayscale = False
            x_hat = np.transpose(x_hat, (0, 2, 3, 1))
            x = np.transpose(x, (0, 2, 3, 1))
        else:
            raise Exception('Invalid number of channels.')

        fig, ax = plt.subplots(2, n, figsize=(n * 3.5, 2.2 + 2 * 3.5))

        for i in range(ax.shape[1]):
            original = x[i]
            reconstructed = x_hat[i]

            for j, im in enumerate((original, reconstructed)):
                axis = ax[j, i]
                if grayscale:
                    axis.imshow(im, cmap='Greys_r')
                else:
                    axis.imshow(im)

                axis.set_xticks([])
                axis.set_yticks([])

        if title is not None:
            fig.suptitle(title, fontsize=40)
        fig.tight_layout()

        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure=plt, figure_name=title)
            plt.clf()
        else:
            plt.show()

    def view_surface_rec(self, x, y, n_max=1000, random_state=42, title=None, dataset_name=None):
        """View 3D original surface and reconstruction.

        Only call this method on 3D surface datasets. x is expected to be 2D.
        Will show figure or log it to Comet if self.comet_exp was set.

        Args:
            x(ndarray): Dataset to sample from.
            y(ndarray): Target variable for coloring.
            n_max(int): Number of points to display.
            random_state(int): Seed for sampling.
            title(str): Figure title.
            dataset_name(str): Dataset name to set customized tilt and rotations.

        """
        if self.comet_exp is not None:
            # If comet_exp is set, use different backend to avoid display errors on clusters
            matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
        import matplotlib.pyplot as plt
        from grae.data.manifolds import set_axes_equal

        np.random.seed(random_state)

        x_hat = self.reconstruct(x)

        if x.shape[0] > n_max:
            sample_mask = np.random.choice(x.shape[0], size=n_max, replace=False)
            x_hat = x_hat[sample_mask]
            x = x[sample_mask]
            y = y[sample_mask]

        scene_dict = dict(SwissRoll=(0, 0), Mammoth=(-15, 90), ToroidalHelices=(30, 0))
        if dataset_name in scene_dict:
            tilt, rotation = scene_dict[dataset_name]
        else:
            tilt, rotation = 0, 0

        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.view_init(tilt, rotation)
        ax.set_title('Input')
        ax.scatter(*x.T, c=y, cmap='jet', edgecolor='k')
        set_axes_equal(ax)

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        ax.view_init(tilt, rotation)
        ax.set_title('Reconstruction')
        ax.scatter(*x_hat.T, c=y, cmap='jet', edgecolor='k')
        set_axes_equal(ax)


        if title is not None:
            fig.suptitle(title, fontsize=20)

        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure=plt, figure_name=title)
            plt.clf()
        else:
            plt.show()
            
"""Torch modules."""
class LinearBlock(nn.Module):
    """FC layer with Relu activation."""

    def __init__(self, in_dim, out_dim):
        """Init.

        Args:
            in_dim(int): Input dimension.
            out_dim(int): Output dimension
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        return F.relu(self.linear(x))

class MLP(nn.Sequential):
    """Sequence of FC layers with Relu activations.

    No activation on last layer, unless sigmoid is requested."""

    def __init__(self, dim_list, sigmoid=False):
        """Init.

        Args:
            dim_list(List[int]): List of dimensions. Ex: [200, 100, 50] will create two layers (200x100 followed by
            100x50).
        """
        # Activations on all layers except last one
        modules = [LinearBlock(dim_list[i - 1], dim_list[i]) for i in range(1, len(dim_list) - 1)]
        modules.append(nn.Linear(dim_list[-2], dim_list[-1]))

        if sigmoid:
            modules.append(nn.Sigmoid())

        super().__init__(*modules)

class AutoencoderModule(nn.Module):
    """Vanilla Autoencoder torch module"""

    def __init__(self, input_dim, hidden_dims, z_dim, noise=0, vae=False, sigmoid=False):
        """Init.

        Args:
            input_dim(int): Dimension of the input data.
            hidden_dims(List[int]): List of hidden dimensions. Do not include dimensions of the input layer and the
            bottleneck. See MLP for example.
            z_dim(int): Bottleneck dimension.
            noise(float): Variance of the gaussian noise applied to the latent space before reconstruction.
            vae(bool): Make this architecture a VAE. Uses an isotropic Gaussian with identity covariance matrix as the
            prior.
            sigmoid(bool): Apply sigmoid to the output.
        """
        super().__init__()
        self.vae = vae

        # Double the size of the latent space if vae to model both mu and logvar
        full_list = [input_dim] + list(hidden_dims) + [z_dim * 2 if vae else z_dim]

        self.encoder = MLP(dim_list=full_list)

        full_list.reverse()  # Use reversed architecture for decoder
        full_list[0] = z_dim

        self.decoder = MLP(dim_list=full_list, sigmoid=sigmoid)
        self.noise = noise

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            tuple:
                torch.Tensor: Reconstructions
                torch.Tensor: Embedding (latent space coordinates)

        """
        z = self.encoder(x)

        # Old idea to inject noise in latent space. Currently not used.
        if self.noise > 0:
            z_decoder = z + self.noise * torch.randn_like(z)
        else:
            z_decoder = z

        if self.vae:
            mu, logvar = z.chunk(2, dim=-1)

            # Reparametrization trick
            if self.training:
                z_decoder = mu + torch.exp(logvar / 2.) * torch.randn_like(logvar)
            else:
                z_decoder = mu

        output = self.decoder(z_decoder)

        # Standard Autoencoder forward pass
        # Note : will still return mu and logvar as a single tensor for compatibility with other classes
        return output, z

# Convolution architecture
class DownConvBlock(nn.Module):
    """Convolutional block.

    3x3 kernel with 1 padding, Max pooling and Relu activations. Channels must be specified by user.

    """
    def __init__(self, in_channels, out_channels):
        """Init.

        Args:
            in_channels(int): Input channels.
            out_channels(int): Output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max = nn.MaxPool2d(2)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.conv(x)
        x = F.relu(x)
        x = self.max(x)

        return x

class UpConvBlock(nn.Module):
    """Transpose convolutional block to upscale input.

    2x2 Transpoe convolution followed by a convolutional layer with
    3x3 kernel with 1 padding, Max pooling and Relu activations. Channels must be specified by user.
    """

    def __init__(self, in_channels, out_channels):
        """Init.

        Args:
            in_channels(int): Input channels.
            out_channels(int): Output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.up(x)
        x = self.conv(x)
        x = F.relu(x)

        return x

class LastConv(UpConvBlock):
    """Add one convolution to UpConvBlock with no activation and kernel_size = 1.

    Used as the output layer in the convolutional AE architecture."""

    def __init__(self, in_channels, out_channels):
        """Init.

        Args:
            in_channels(int): Input channels.
            out_channels(int): Output channels.
        """
        super().__init__(in_channels, in_channels)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = super().forward(x)
        x = self.conv_2(x)

        return x

class ConvEncoder(nn.Module):
    """Convolutional encoder for images datasets.

    Convolutions (with 3x3 kernel size, see DownConvBlock for details) followed by a FC network."""
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim):
        """Init.

        Args:
            H(int): Height of the input data.
            W(int): Width of the input data
            input_channel(int): Number of channels in the input data. Typically 1 for grayscale and 3 for RGB.
            channel_list(List[int]): List of channels. Determines the number of convolutional layers and associated
            channels.  ex: [128, 64, 32] defines one layer with in_channels=128 and out_channels=64 followed by a
            second one with in_channels=64 and out_channels=32.
            hidden_dims(List[int]): List of hidden dimensions for the FC network after the convolutions.
            Do not include dimensions of the input layer and the bottleneck. See MLP for example.
            z_dim(int): Dimension of the bottleneck.
        """
        super().__init__()

        channels = [input_channel] + channel_list
        modules = list()

        for i in range(1, len(channels)):
            modules.append(DownConvBlock(in_channels=channels[i - 1], out_channels=channels[i]))

        self.conv = nn.Sequential(*modules)

        factor = 2 ** len(channel_list)

        self.fc_size = int(channel_list[-1] * H / factor * W / factor)  # Compute size of FC input

        mlp_dim = [self.fc_size] + hidden_dims + [z_dim]

        self.linear = MLP(mlp_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.conv(x)
        x = x.view(-1, self.fc_size)
        x = self.linear(x)

        return x

class ConvDecoder(nn.Module):
    """Convolutional decoder for images datasets.

    FC architecture followed by upscaling convolutions.
    Note that last layer uses a 1x1 kernel with no activations. See UpConvBlock and LastConv for details."""
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim, sigmoid):
        """Init.

        Args:
            H(int): Height of the reconstructed data.
            W(int): Width of the reconstructed data
            input_channel(int): Number of channels in the reconstructed data. Typically 1 for grayscale and 3 for RGB.
            channel_list(List[int]): List of channels. Determines the number of UpConvBlock and associated
            channels.  ex: [32, 64, 128] defines one layer with in_channels=32 and out_channels=64 followed by a
            second one with in_channels=64 and out_channels=128.
            hidden_dims(List[int]): List of hidden dimensions for the FC network before the convolutions.
            Do not include dimensions of the input layer and the bottleneck. See MLP for example.
            z_dim(int): Dimension of the bottleneck.
            sigmoid(bool) : Apply sigmoid to output.
        """
        super().__init__()
        self.H = H
        self.W = W

        self.factor = 2 ** len(channel_list)

        fc_size = int(channel_list[0] * H / self.factor * W / self.factor)

        mlp_dim = [z_dim] + hidden_dims + [fc_size]

        self.linear = MLP(mlp_dim)

        channels = channel_list
        modules = list()

        for i in range(1, len(channels)):
            modules.append(UpConvBlock(in_channels=channels[i - 1], out_channels=channels[i]))

        modules.append(LastConv(in_channels=channels[-1], out_channels=input_channel))

        if sigmoid:
            modules.append(nn.Sigmoid())

        self.conv = nn.Sequential(*modules)
        self.first_channel = channel_list[0]

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.linear(x)
        x = x.view(-1, self.first_channel, self.H // self.factor, self.W // self.factor)
        x = self.conv(x)

        return x

class ConvAutoencoderModule(nn.Module):
    """Autoencoder with convolutions for image datasets."""
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim, noise, vae=False, sigmoid=False):
        """Init. Arguments specify the architecture of the encoder. Decoder will use the reverse architecture.

        Args:
            H(int): Height of the input data.
            W(int): Width of the input data
            input_channel(int): Number of channels in the input data. Typically 1 for grayscale and 3 for RGB.
            channel_list(List[int]): List of channels. Determines the number of convolutional layers and associated
            channels.  ex: [128, 64, 32] defines one layer with in_channels=128 and out_channels=64 followed by a
            second one with in_channels=64 and out_channels=32.
            hidden_dims(List[int]): List of hidden dimensions for the FC network after the convolutions.
            Do not include dimensions of the input layer and the bottleneck. See MLP for example.
            z_dim(int): Dimension of the bottleneck.
            noise(float): Variance of the gaussian noise applied to the latent space before reconstruction.
            vae(bool): Make this architecture a VAE. Uses an isotropic Gaussian with identity covariance matrix as the
            prior.
            sigmoid(bool): Apply sigmoid to the output.
        """
        super().__init__()
        self.vae = vae

        # Double size of encoder output if using a VAE to model both mu and logvar
        self.encoder = ConvEncoder(H, W, input_channel, channel_list, hidden_dims, z_dim * 2 if self.vae else z_dim)
        channel_list.reverse()
        hidden_dims.reverse()
        self.decoder = ConvDecoder(H, W, input_channel, channel_list, hidden_dims, z_dim, sigmoid)
        self.noise = noise

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            tuple:
                torch.Tensor: Reconstructions
                torch.Tensor: Embedding (latent space coordinates)
        """
        # Same forward pass as standard autoencoder
        return AutoencoderModule.forward(self, x)
    
"""AE and GRAE model classes with sklearn inspired interface."""

class AE(BaseModel):
    """Vanilla Autoencoder model.

    Trained with Adam and MSE Loss.
    Model will infer from the data whether to use a fully FC or convolutional + FC architecture.
    """

    def __init__(self, *,
                 lr=LR,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 weight_decay=WEIGHT_DECAY,
                 random_state=SEED,
                 n_components=2,
                 hidden_dims=HIDDEN_DIMS,
                 conv_dims=CONV_DIMS,
                 conv_fc_dims=CONV_FC_DIMS,
                 noise=0,
                 patience=50,
                 data_val=None,
                 comet_exp=None,
                 write_path='', 
                 device = DEVICE):
        """Init. Arguments specify the architecture of the encoder. Decoder will use the reversed architecture.

        Args:
            lr(float): Learning rate.
            epochs(int): Number of epochs for model training.
            batch_size(int): Mini-batch size.
            weight_decay(float): L2 penalty.
            random_state(int): To seed parameters and training routine for reproducible results.
            n_components(int): Bottleneck dimension.
            hidden_dims(List[int]): Number and size of fully connected layers for encoder. Do not specify the input
            layer or the bottleneck layer, since they are inferred from the data or from the n_components
            argument respectively. Decoder will use the same dimensions in reverse order. This argument is only used if
            provided samples are flat vectors.
            conv_dims(List[int]): Specify the number of convolutional layers. The int values specify the number of
            channels for each layer. This argument is only used if provided samples are images (i.e. 3D tensors)
            conv_fc_dims(List[int]): Number and size of fully connected layers following the conv_dims convolutionnal
            layer. No need to specify the bottleneck layer. This argument is only used if provided samples
            are images (i.e. 3D tensors)
            noise(float): Variance of the gaussian noise injected in the bottleneck before reconstruction.
            patience(int): Epochs with no validation MSE improvement before early stopping.
            data_val(BaseDataset): Split to validate MSE on for early stopping.
            comet_exp(Experiment): Comet experiment to log results.
            write_path(str): Where to write temp files.
        """
        self.random_state = random_state
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.fitted = False  # If model was fitted
        self.torch_module = None  # Will be initialized to the appropriate torch module when fit method is called
        self.optimizer = None  # Will be initialized to the appropriate optimizer when fit method is called
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss(reduction='mean')
        self.conv_dims = conv_dims
        self.conv_fc_dims = conv_fc_dims
        self.noise = noise
        self.comet_exp = comet_exp
        self.data_shape = None  # Shape of input data
        self.device = device

        # Early stopping attributes
        self.data_val = data_val
        self.val_loader = None
        self.patience = patience
        self.current_loss_min = np.inf
        self.early_stopping_count = 0
        self.write_path = write_path

    def init_torch_module(self, data_shape, vae=False, sigmoid=False):
        """Infer autoencoder architecture (MLP or Convolutional + MLP) from data shape.

        Initialize torch module.

        Args:
            data_shape(tuple[int]): Shape of one sample.
            vae(bool): Make this architecture a VAE.
            sigmoid(bool): Apply sigmoid to decoder output.

        """
        # Infer input size from data. Initialize torch module and optimizer
        if len(data_shape) == 1:
            # Samples are flat vectors. MLP case
            input_size = data_shape[0]
            self.torch_module = AutoencoderModule(input_dim=input_size,
                                                  hidden_dims=self.hidden_dims,
                                                  z_dim=self.n_components,
                                                  noise=self.noise,
                                                  vae=vae,
                                                  sigmoid=sigmoid)
            
        elif len(data_shape) == 2:
            # Treat 2D data as a single-channel image
            in_channel = 1
            height, width = data_shape
            self.torch_module = ConvAutoencoderModule(H=height,
                                  W=width,
                                  input_channel=in_channel,
                                  channel_list=self.conv_dims,
                                  hidden_dims=self.conv_fc_dims,
                                  z_dim=self.n_components,
                                  noise=self.noise,
                                  vae=vae,
                                  sigmoid=sigmoid)
            
        elif len(data_shape) == 3:
            in_channel, height, width = data_shape
            #  Samples are 3D tensors (i.e. images). Convolutional case.
            self.torch_module = ConvAutoencoderModule(H=height,
                                                      W=width,
                                                      input_channel=in_channel,
                                                      channel_list=self.conv_dims,
                                                      hidden_dims=self.conv_fc_dims,
                                                      z_dim=self.n_components,
                                                      noise=self.noise,
                                                      vae=vae,
                                                      sigmoid=sigmoid)
        else:
            raise Exception(f'Invalid channel number. X has {len(data_shape)}')

        self.torch_module.to(self.device)

    def fit(self, x):
        """Fit model to data.

        Args:
            x(ndarray or Torch Tensor): Dataset to fit.

        """

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Reproducibility
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Save data shape
        self.data_shape = x[0].shape #EDITED TODO: May need fixing since changing x data type to torch tensor

        # Fetch appropriate torch module
        if self.torch_module is None:
            self.init_torch_module(self.data_shape)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        # Train AE
        # Training steps are decomposed as calls to specific methods that can be overriden by children class if need be
        self.torch_module.train()

        self.loader = self.get_loader(x)

        if self.data_val is not None:
            self.val_loader = self.get_loader(self.data_val)

        # Get first metrics
        self.log_metrics(0)

        for epoch in range(1, self.epochs + 1):
            # print(f'            Epoch {epoch}...')
            for batch in self.loader:
                self.optimizer.zero_grad()
                self.train_body(batch)
                self.optimizer.step()

            self.log_metrics(epoch)
            self.end_epoch(epoch)

            # Early stopping
            if self.early_stopping_count == self.patience:
                if self.comet_exp is not None:
                    self.comet_exp.log_metric('early_stopped',
                                              epoch - self.early_stopping_count)
                break

        # Load checkpoint if it exists
        checkpoint_path = os.path.join(self.write_path, 'checkpoint.pt')

        if os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
            os.remove(checkpoint_path)

    def get_loader(self, x):
        """Fetch data loader.

        Args:
            x(ndarray or Torch Tensor): Data to be wrapped in loader.

        Returns:
            torch.utils.data.DataLoader: Torch DataLoader for mini-batch training.

        """
        if isinstance(x, torch.Tensor):
            dataset = FromNumpyDataset(x.numpy())
        else:
            dataset = FromNumpyDataset(x)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_body(self, batch):
        """Called in main training loop to update torch_module parameters.

        Args:
            batch(tuple[torch.Tensor]): Training batch.

        """
        data, _, idx = batch  # No need for labels. Training is unsupervised
        data = data.to(self.device)

        x_hat, z = self.torch_module(data)  # Forward pass
        self.compute_loss(data, x_hat, z, idx)

    def compute_loss(self, x, x_hat, z, idx):
        """Apply loss to update parameters following a forward pass.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        loss = self.criterion(x_hat, x)
        loss.backward()

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Args:
            epoch(int): Current epoch.

        """
        pass

    def eval_MSE(self, loader):
        """Compute MSE on data.

        Args:
            loader(DataLoader): Dataset loader.

        Returns:
            float: MSE.

        """
        # Compute MSE over dataset in loader
        self.torch_module.eval()
        sum_loss = 0

        for batch in loader:
            data, _, idx = batch  # No need for labels. Training is unsupervised
            data = data.to(self.device)

            x_hat, z = self.torch_module(data)  # Forward pass
            sum_loss += data.shape[0] * self.criterion(data, x_hat).item()

        self.torch_module.train()

        return sum_loss / len(loader.dataset)  # Return average per observation

    def log_metrics(self, epoch):
        """Log metrics.

        Args:
            epoch(int): Current epoch.

        """
        self.log_metrics_train(epoch)
        self.log_metrics_val(epoch)

    def log_metrics_val(self, epoch):
        """Compute validation metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Validation loss
        if self.val_loader is not None:
            val_mse = self.eval_MSE(self.val_loader)

            if self.comet_exp is not None:
                with self.comet_exp.validate():
                    self.comet_exp.log_metric('MSE_loss', val_mse, epoch=epoch)

            if val_mse < self.current_loss_min:
                # If new min, update attributes and checkpoint model
                self.current_loss_min = val_mse
                self.early_stopping_count = 0
                self.save(os.path.join(self.write_path, 'checkpoint.pt'))
            else:
                self.early_stopping_count += 1

    def log_metrics_train(self, epoch):
        """Log train metrics, log them to comet if need be and update early stopping attributes.

        Args:
            epoch(int):  Current epoch.
        """
        # Train loss
        if self.comet_exp is not None:
            train_mse = self.eval_MSE(self.loader)
            with self.comet_exp.train():
                self.comet_exp.log_metric('MSE_loss', train_mse, epoch=epoch)

    def transform(self, x):
        """Transform data.

        Args:
            x(ndarray): Dataset to transform.
        Returns:
            ndarray: Embedding of x.

        """
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        z = [self.torch_module.encoder(batch.to(self.device)).cpu().detach().numpy() for batch, _, _ in loader]
        return np.concatenate(z)

    def inverse_transform(self, x):
        """Take coordinates in the embedding space and invert them to the data space.

        Args:
            x(ndarray): Points in the embedded space with samples on the first axis.
        Returns:
            ndarray: Inverse (reconstruction) of x.

        """
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = torch.utils.data.DataLoader(x, batch_size=self.batch_size,
                                             shuffle=False)
        x_hat = [self.torch_module.decoder(batch.to(self.device)).cpu().detach().numpy()
                 for batch in loader]

        return np.concatenate(x_hat)

    def save(self, path):
        """Save state dict.

        Args:
            path(str): File path.

        """
        state = self.torch_module.state_dict()
        state['data_shape'] = self.data_shape
        torch.save(state, path)

    def load(self, path):
        """Load state dict.

        Args:
            path(str): File path.

        """
        state = torch.load(path)
        data_shape = state.pop('data_shape')

        if self.torch_module is None:
            self.init_torch_module(data_shape)

        self.torch_module.load_state_dict(state)

class GRAEBase(AE):
    """Geometric Regularized Autoencoder specific to work for precomputed aligned manifolds.

    AE with geometry regularization. The bottleneck is regularized to match an embedding precomputed by a manifold
    learning algorithm.
    """

    def __init__(self, lam=100, relax=False, device = DEVICE, **kwargs):
        """Init.

        Args:
            lam(float): Regularization factor.
            relax(bool): Use the lambda relaxation scheme. Set to false to use constant lambda throughout training.
            device(str): Device to use for training.
            **kwargs: All other arguments with keys are passed to the AE parent class.
        """
        super().__init__(device = device, **kwargs)
        self.lam = lam
        self.lam_original = lam  
        self.target_embedding = None 
        self.relax = relax
        self.device = device
        self.loss = 0
        self.history = {"epoch": [], "loss": []}  


    def fit(self, x, emb, verbose = 0):
        """Fit model to data.

        Args:
            x(ndarray): Dataset to fit.

        """
        if verbose != 0:
            print('       Fitting GRAE...')
            print('           Fitting manifold learning embedding...')
        #emb = scipy.stats.zscore(self.embedder.fit_transform(x))  # Normalize embedding
        self.target_embedding = torch.from_numpy(emb).float().to(self.device)

        if verbose != 0:
            print('           Fitting encoder & decoder...')
        super().fit(x)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute torch-compatible geometric loss.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        if self.lam > 0:
            loss = self.criterion(x, x_hat) + self.lam * self.criterion(z, self.target_embedding[idx])
        else:
            loss = self.criterion(x, x_hat)

        loss.backward()
        self.loss += loss.item()

    def log_metrics_train(self, epoch):
        """Log train metrics to comet if comet experiment was set.

        Args:
            epoch(int): Current epoch.

        """
        if self.comet_exp is not None:

            # Compute MSE and Geometric Loss over train set
            self.torch_module.eval()
            sum_loss = 0
            sum_geo_loss = 0

            for batch in self.loader:
                data, _, idx = batch  # No need for labels. Training is unsupervised
                data = data.to(self.device)

                x_hat, z = self.torch_module(data)  # Forward pass
                sum_loss += data.shape[0] * self.criterion(data, x_hat).item()
                sum_geo_loss += data.shape[0] * self.criterion(z, self.target_embedding[idx]).item()

            with self.comet_exp.train():
                mse_loss = sum_loss / len(self.loader.dataset)
                geo_loss = sum_geo_loss / len(self.loader.dataset)
                self.comet_exp.log_metric('MSE_loss', mse_loss, epoch=epoch)
                self.comet_exp.log_metric('geo_loss', geo_loss, epoch=epoch)
                self.comet_exp.log_metric('GRAE_loss', mse_loss + self.lam * geo_loss, epoch=epoch)
                if self.lam * geo_loss > 0:
                    self.comet_exp.log_metric('geo_on_MSE', self.lam * geo_loss / mse_loss, epoch=epoch)

            self.torch_module.train()

    def update_history(self, epoch):
        """Log loss history for each epoch."""
        self.history["epoch"].append(epoch)
        self.history["loss"].append(self.loss)
        self.loss = 0

    def end_epoch(self, epoch):
        """Method called at the end of every training epoch.

        Args:
            epoch(int): Current epoch.
        """
        self.update_history(epoch)

        if self.relax and self.lam > 0 and self.early_stopping_count == int(self.patience / 2):
            self.lam = 0  # Turn off constraint

            if self.comet_exp is not None:
                self.comet_exp.log_metric('relaxation', epoch, epoch=epoch)

class anchorGRAE(GRAEBase):
    """GRAE but with anchor loss applied in the embedding space.

    This class extends the GRAEBase class to include an additional anchor loss term in the embedding space.
    The anchor loss ensures that specific anchor points from domain A are mapped to corresponding points in domain B.
    """

    def __init__(self, anchor_lam=100, **kwargs):
        """
        Initializes the AutoEncoder with the given parameters.
        Args:
            anchor_lam (int, optional): The lambda value for the anchor. Default is 100.
            **kwargs: Additional keyword arguments to pass to the superclass initializer.
        """

        super().__init__(**kwargs)
        self.anchor_lam = anchor_lam

    def fit(self, A, emb, anchors, verbose = 0):
        """
         Args:
            A (ndarray): Data from domain A.
            emb (ndarray): Precomputed embeddings for A.
            anchors (ndarray): Known anchor points.

        Anchors need to be so that the encoder is first, then Decoder second.
        """
        #Save the anchors
        self.anchors = anchors 
        
        super().fit(A, emb, verbose)

    def compute_loss(self, x, x_hat, z, idx):
        """Compute torch-compatible geometric loss.

        Args:
            x(torch.Tensor): Input batch.
            x_hat(torch.Tensor): Reconstructed batch (decoder output).
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        if self.lam > 0:
            loss = self.criterion(x, x_hat) + self.lam * self.criterion(z, self.target_embedding[idx])
        else:
            loss = self.criterion(x, x_hat)

        #Create a subset of the idexes that are also anchors so we can compare anchor to anchor
        anchor_idx = [i for i in range(len(idx)) if idx[i] in self.anchors[:, 0]]

        # Domain Translation loss - We only want to do this if its an anchor point!!!
        if self.anchor_lam > 0 and len(anchor_idx) > 0:
            loss += self.criterion(self.target_embedding[idx[anchor_idx]], z[anchor_idx]) * self.anchor_lam # I think we should weight this one the most?


        loss.backward()
        self.loss += loss.item()

class DomainTranslation():
    """
    NOTE: This is original to Adam. 

    Custom domain translation class with custom loss functions.
    
    - Loss from A to Z layer back to A (Standard)
    - Loss from anchors in A to Z layer to B (Weighted)
    - Loss from points from A to Z layer to B domain back to Z to A
    """

    def __init__(self, A_lam=100, A_relax=False, Akwargs={}, B_lam=100, B_relax=False, Bkwargs={}, 
                 anchor_weight=1.0, cycle_weight=1.0):
        """
        Args:
            A_lam, A_relax, Akwargs: Parameters for GRAE A.
            B_lam, B_relax, Bkwargs: Parameters for GRAE B.
            anchor_weight: Weight for anchor loss.
            cycle_weight: Weight for cycle consistency loss.
        """
        self.graeA = GRAEBase(A_lam, A_relax, **Akwargs)
        self.graeB = GRAEBase(B_lam, B_relax, **Bkwargs)
        self.anchor_weight = anchor_weight
        self.cycle_weight = cycle_weight

    def compute_custom_loss(self, A, B, Z_A, Z_B, idx): #    def compute_loss(self, x, x_hat, z, idx):

        """
        Compute the combined loss for domain translation.
        
        Args:
            A (torch.Tensor): Original data from domain A.
            B (torch.Tensor): Original data from domain B.
            Z_A (torch.Tensor): Latent representation of A (from graeA).
            Z_B (torch.Tensor): Latent representation of B (from graeB).
            idx_A (torch.Tensor): Indices for A samples.
            idx_B (torch.Tensor): Indices for B samples.
        
        Returns:
            torch.Tensor: Combined loss value.
        """
        # Standard reconstruction loss (A -> Z -> A)
        loss_A = self.graeA.criterion(A, torch.tensor(self.graeA.inverse_transform(Z_A), device=DEVICE, requires_grad=True)) 

        # Anchor loss (A -> Z -> B embedding)
        A_Z_B_data = self.graeB.inverse_transform(Z_A)
        anchor_loss_A = self.anchor_weight * self.graeB.criterion(torch.tensor(A_Z_B_data[idx], device=DEVICE, requires_grad=True), B[idx])

        # Cycle consistency loss (A -> Z -> B -> Z -> A)
        A_Z_B_data = BaseDataset(x = A_Z_B_data, y = np.zeros(len(A_Z_B_data)), split_ratio = 0.8, random_state = 42, split = "none")
        A_reconstructed = self.graeA.inverse_transform(self.graeB.transform(A_Z_B_data))  # -> Z -> A
        cycle_loss_A = self.cycle_weight * self.graeA.criterion(A, torch.tensor(A_reconstructed, device=DEVICE, requires_grad=True))

        """DO the Same from B's perspective"""
        # loss_B = self.graeB.criterion(B, self.graeB.inverse_transform(Z_B)) #NOTE: Do we want to calculate the loss from both perspectives? 

        # # Anchor loss (B -> Z -> A embedding)
        # B_Z_A_data = self.graeA.inverse_transform(Z_B)
        # anchor_loss_B = self.anchor_weight * self.graeA.criterion(B_Z_A_data[idx], B[idx])

        # # Cycle consistency loss (B -> Z -> A -> Z -> B)
        # B_reconstructed = self.graeB.inverse_transform(self.graeA.transform(B_Z_A_data))  # -> Z -> B
        # cycle_loss_B = self.cycle_weight * self.graeB.criterion(B, B_reconstructed)

        if anchor_loss_A > 0:
            return loss_A + anchor_loss_A * 2 + cycle_loss_A #+ loss_B + anchor_loss_B + cycle_loss_B
        else:
            return loss_A + cycle_loss_A

    def custom_collate_fn(self, batch):
        A_batch = np.stack([item[0] for item in batch])
        B_batch = np.stack([item[1] for item in batch])
        is_anchor = np.array([item[2] for item in batch])
        return A_batch, B_batch, is_anchor
    
    def fit(self, A, B, labels, emb, known_anchors, epochs):
        """
        Fit model to data from domains A and B.

        Args:
            A (torch.Tensor): Data from domain A.
            B (torch.Tensor): Data from domain B.
            emb_A (torch.Tensor): Precomputed embeddings for A.
            emb_B (torch.Tensor): Precomputed embeddings for B.
        """
        print('Fitting GRAE modules...')
        dataset_A = BaseDataset(x = A, y = labels, split_ratio = 0.8, random_state = 42, split = "none")
        dataset_B = BaseDataset(x = B, y = labels, split_ratio = 0.8, random_state = 42, split = "none")
        self.graeA.fit(dataset_A, emb[:len(A)])
        self.graeB.fit(dataset_B, emb[len(A):])


        print("\nPreparing Anchor Data...")
        #How to handle Anchors. 
        #1. Batch Data A and B together so they points remained connected
        #2. Flag which ones are anchors, and create the indexes after that. 

        tupled_data = []
        for pair in known_anchors:
            tupled_data.append((A[pair[0]], B[pair[1]], True))
        
        # Collect indices to be deleted
        indices_A = [pair[0] for pair in known_anchors]
        indices_B = [pair[1] for pair in known_anchors]

        # Delete all collected indices at once
        A = np.delete(A, indices_A, axis=0)
        B = np.delete(B, indices_B, axis=0)

        small_data_size = min(A.shape[0], B.shape[0])
        for i in range(0, small_data_size):
            tupled_data.append((A[i], B[i], False))
        
        loader = torch.utils.data.DataLoader(tupled_data, batch_size=32, shuffle=True, collate_fn=self.custom_collate_fn)

        # Move graeA and graeB parameters to the correct device
        self.graeA.torch_module.to(DEVICE)
        self.graeB.torch_module.to(DEVICE)

        # Initialize the optimizer after moving parameters to the correct device
        self.optimizer = torch.optim.Adam(
            list(self.graeA.torch_module.parameters()) + list(self.graeB.torch_module.parameters()),
            lr=self.graeA.lr * 0.1, #Fine tuning
            weight_decay=self.graeA.weight_decay
        )

        print('\n ---------------------------------\nBeginning Training Loop...')
        for epoch in range(epochs):
            for batch in loader:
                A_batch, B_batch, is_anchor = batch

                idx = np.where(is_anchor)

                # Convert A_batch and B_batch to BaseDataset
                dataset_A_batch = BaseDataset(x=A_batch, y=np.zeros(len(A_batch)), split_ratio=0.8, random_state=42, split="none")
                dataset_B_batch = BaseDataset(x=B_batch, y=np.zeros(len(B_batch)), split_ratio=0.8, random_state=42, split="none")

                # Forward pass through GRAEs
                Z_A = self.graeA.transform(dataset_A_batch)
                Z_B = self.graeB.transform(dataset_B_batch)

                # Compute custom loss
                A_batch = torch.tensor(A_batch, dtype=torch.float32, device=DEVICE, requires_grad=True)
                B_batch = torch.tensor(B_batch, dtype=torch.float32, device=DEVICE, requires_grad=True)
                loss = self.compute_custom_loss(A_batch, B_batch, Z_A, Z_B, idx)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
# Define a custom module that chains encoderA -> decoderB -> encoderB -> decoderA
class SwappedModule(nn.Module):
    def __init__(self, encoderA, decoderB, encoderB, decoderA, vae=False):
        super().__init__()
        self.encoderA = encoderA
        self.decoderB = decoderB
        self.encoderB = encoderB
        self.decoderA = decoderA
        self.vae = vae

     #Overide from the Torch_Modules -> Note this is the same despite the archetecture differences
    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            tuple:
                torch.Tensor: Reconstructions
                torch.Tensor: Embedding (latent space coordinates)

        """
        # A to Z
        a_z = self.encoderA(x)
        a_z_2 = a_z

        if self.vae:
            mu, logvar = a_z.chunk(2, dim=-1)

            # Reparametrization trick
            if self.training:
                a_z_2 = mu + torch.exp(logvar / 2.) * torch.randn_like(logvar)
            else:
                a_z_2 = mu

        #Z to B
        b = self.decoderB(a_z_2)

        # B to Z
        b_z = self.encoderB(b)
        b_z_2 = b_z

        if self.vae:
            mu, logvar = b_z.chunk(2, dim=-1)

            # Reparametrization trick
            if self.training:
                b_z_2 = mu + torch.exp(logvar / 2.) * torch.randn_like(logvar)
            else:
                b_z_2 = mu

        # Z to A
        a = self.decoderA(b_z_2)

        # Standard Autoencoder forward pass
        return a, b, a_z, b_z
    
class SwappedGRAE(GRAEBase):
    """Helper Class that swaps the encoder and decoder of a GRAE model."""
    def __init__(self, encoderA, decoderA, encoderB, decoderB, lam_A_to_B = 2, lam_A_to_A = 1, **kwargs):
        #Create a new device to complete the secondary training
        self.device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")

        self.lam_A_to_B = lam_A_to_B
        self.lam_A_to_A = lam_A_to_A

        super().__init__(device = self.device, **kwargs)

        #This represents the full loop expressed as A -> Z -> B -> Z -> A
        self.encoderA = encoderA.to(self.device)
        self.decoderB = decoderB.to(self.device)
        self.encoderB = encoderB.to(self.device)
        self.decoderA = decoderA.to(self.device)

        # Instantiate the module with the collected components.
        self.torch_module = SwappedModule(self.encoderA, self.decoderB, self.encoderB, self.decoderA, vae=getattr(self, "vae", False))

    def fit(self, A, B,  emb, anchors):
        """Fit model to data.
        
        Anchors need to be so that the encoder is first, then Decoder second"""
        #Save the anchors
        self.anchors = anchors #NOTE: Instead of using the full dataset, I should just use the anchors as the data. I can keep the anchors to check the indicies. I also could keep twin GRAE swaps

        #Save Data B
        self.B = B.data.to(self.device) # We have to index it because its a BaseDataset object

        super().fit(A, emb)

    #Overide from GRAE
    def compute_loss(self, A, a, b, a_z, b_z, idx):
        """Compute torch-compatible geometric loss.

        Args:
            x(torch.Tensor): Input batch -> Data from domain A (Or B)
            x_hat(torch.Tensor): Reconstructed batch (decoder output). -> Data from domain B (or A)
            z(torch.Tensor): Batch embedding (encoder output).
            idx(torch.Tensor): Indices of samples in batch.

        """
        #Create a subset of the idexes that are also anchors so we can compare anchor to anchor
        anchor_idx = [i for i in range(len(idx)) if idx[i] in self.anchors[:, 0]]

        #Full Reconstruction loss (A to A)
        if self.lam_A_to_A > 0:
            loss = self.criterion(A, a) * self.lam_A_to_A
        else:
            loss = torch.tensor(0.0, device=A.device, requires_grad=True)

        # Domain Translation loss - We only want to do this if its an anchor point!!!
        if self.lam_A_to_B > 0 and len(anchor_idx) > 0:
            loss += self.criterion(self.B[idx[anchor_idx]], b[anchor_idx]) * self.lam_A_to_B # I think we should weight this one the most?

        #Geometry Regularization to the Embedding
        if self.lam > 0:
            #Embedding loss (A to z and B to z)
            loss += self.lam * self.criterion(a_z, self.target_embedding[idx])
            loss += self.lam * self.criterion(b_z, self.target_embedding[idx])

        loss.backward()
        self.loss += loss.item()

    #Override from GRAE's parent AE
    def train_body(self, batch):
        """Called in main training loop to update torch_module parameters.

        Args:
            batch(tuple[torch.Tensor]): Training batch.

        """
        data, _, idx = batch  # No need for labels. Training is unsupervised
        data = data.to(self.device)

        #TODO: We will want to do this twice
        a, b, a_z, b_z = self.torch_module(data)  # Forward pass

        try:
            self.compute_loss(data, a, b, a_z, b_z, idx)
        except Exception as e:
            raise Exception(f"Error in compute_loss: {e}. CHECK THE DATA YOU ENTERED")
    
    #TODO: Relook the transform and inverse transform functions | We will need a full transform (A to Z to B) and an Inverse
    def transform(self, A):
        "Returns the full process A to Z to B to Z to A. Returns the final A and B."

        self.torch_module.eval()
        with torch.no_grad():
            if not torch.is_tensor(A):
                A = torch.tensor(A, dtype=torch.float32, device=self.device)
            # Full forward pass through the swapped module:
            #   encoderA -> decoderB gives the A-to-B translation
            #   encoderB -> decoderA gives the B-to-A translation (in the cycle)
            a, b, a_z, b_z = self.torch_module(A)

        #Return the predicted A and B
        return a.cpu().numpy(), b.cpu().numpy(), a_z.cpu().numpy(), b_z.cpu().numpy()
        
    def inverse_transform(self, B):
        self.torch_module.eval()
        with torch.no_grad():
            if not torch.is_tensor(B):
                B = torch.tensor(B, dtype=torch.float32, device=self.device)
            # For inverse, we apply the reverse transformation:
            
            # Map from domain B back to A via encoderB then decoderA.
            b_z = self.encoderB(B)
            a = self.decoderA(b_z)
        
            # Map from domain A back to B via encoderA then decoderB.
            a_z = self.encoderA(a)
            b = self.decoderB(b_z)

        return a.cpu().numpy(), b.cpu().numpy(), a_z.cpu().numpy(), b_z.cpu().numpy()

class TAEROE():
    #TODO: Right now I am assuming all the anchors are paired [1, 1] and never [2,1]. Write code later to enforce this. 
    """
    Twin AutoEncoders with Regularization to Observed Embedding (TAEROE) class.
    NOTE: This is original to Adam. 
    """

    #Overide from GRAE
    def __init__(self, A_lam=100, A_relax=False, Akwargs={}, B_lam=100, B_relax=False, Bkwargs={}, 
                 anchor_weight=1.0, cycle_weight=1.0, verbose = 0, epochs = 200, SGkwargs = {}):
        """
        Args:
            A_lam, A_relax, Akwargs: Parameters for GRAE A.
            B_lam, B_relax, Bkwargs: Parameters for GRAE B.
            anchor_weight: Weight for anchor loss.
            cycle_weight: Weight for cycle consistency loss.
        """
        #Create the twin Regularized AutoEncoders
        self.graeA = GRAEBase(A_lam, A_relax, **Akwargs)
        self.graeB = GRAEBase(B_lam, B_relax, **Bkwargs)

        #Set the weights for the anchor and cycle loss
        self.anchor_weight = anchor_weight
        self.cycle_weight = cycle_weight
        self.epochs = epochs
        self.verbose = verbose
        self.SGkwargs = SGkwargs
    
    def fit(self, A, B, emb, known_anchors, labelsA = None, labelsB = None):
        """
        Fit model to data from domains A and B.

        Labels are simply for coloring the graph

        Args:
            A (torch.Tensor): Data from domain A.
            B (torch.Tensor): Data from domain B.
            emb_A (torch.Tensor): Precomputed embeddings for A.
            emb_B (torch.Tensor): Precomputed embeddings for B.
        """
        if self.verbose > 0:
            print('Fitting GRAE modules...')

        #Save the known Anchors
        self.known_anchors = known_anchors

        #Add null labels if labels aren't given
        if labelsA is None:
            labelsA = np.zeros(len(A))
        if labelsB is None:
            labelsB = np.zeros(len(B))
            
        dataset_A = BaseDataset(x = A, y = labelsA, split_ratio = 0.8, random_state = 42, split = "none")
        dataset_B = BaseDataset(x = B, y = labelsB, split_ratio = 0.8, random_state = 42, split = "none")
        self.graeA.fit(dataset_A, emb[:len(A)])
        self.graeB.fit(dataset_B, emb[len(A):])

        #Select the encoders and decoders
        encoderA = self.graeA.torch_module.encoder
        decoderA = self.graeA.torch_module.decoder
        encoderB = self.graeB.torch_module.encoder
        decoderB = self.graeB.torch_module.decoder

        if self.verbose > 0:
            print("\n------------------------------------------------\nBeginning Training Loop for Swapped model...")

        self.swapped = SwappedGRAE(encoderA, decoderA, encoderB, decoderB, **self.SGkwargs)
        self.swapped.fit(dataset_A, dataset_B, emb, known_anchors)

        if self.verbose > 0:
            print("\n Processed Finished.")
    
    def transform(self, A):
        return self.swapped.transform(A)

    def inverse_transform(self, B):
        return self.swapped.inverse_transform(B)
    

    def plot_histories(self, same_plot=True):
        """Plot loss curves history and return the loss values.

        Args:
            same_plot (bool): If True, plot all histories on the same plot.
                              If False, plot each history on separate subplots.

        Returns:
            dict: Dictionary containing loss history with keys 'epoch' and 'loss'.
        """
        import matplotlib.pyplot as plt

        histories = [self.graeA.history, self.graeB.history, self.swapped.history]
        labels = ['GRAE A', 'GRAE B', 'Swapped']
        colors = ['b', 'r', 'g']

        if same_plot:
            plt.figure(figsize=(15, 7))
            for idx, history in enumerate(histories):
                plt.plot(history["epoch"], history["loss"], color=colors[idx],
                         marker='o', label=labels[idx])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for idx, history in enumerate(histories):
                axes[idx].plot(history["epoch"], history["loss"], color=colors[idx],
                               marker='o', label=labels[idx])
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel('Loss')
                axes[idx].set_title(f'{labels[idx]} Training Loss Curve')
                axes[idx].grid(True)
                axes[idx].legend()
            plt.tight_layout()
            plt.show()

        return {"epoch": histories[0]["epoch"], "loss": [history["loss"] for history in histories]}

class TAEROE2():
    """
    Try creating a swapped GRAE except for maintaining two networks -> retrained with anchor loss
    """      

    pass

class EmbeddingProber:
    """Class to benchmark MSE, the coefficient of determination (R2) for ground truth continuous variables and
    classification accuracy of dataset has labels.
    """
    def fit(self, model, dataset, mse_only=False):
        """Fit regressors to predict latent variables and/or labels if available.

        If a dataset has multiple latent variables, one regressor is used per variable. Moreover, if the dataset has
        latent variables in addition to class labels, the data is divided according to labels and one regressor
        is trained per combination of label/latent variable.

        Args:
            model(BaseModel): Fitted Model.
            dataset(BaseDataset): Dataset to benchmark.
            mse_only(optional, bool): Compute only MSE. Useful for lightweight computations during hyperparameter search.

        """
        self.linear_regressors = []
        self.linear_classifiers = []
        self.mse_only = mse_only

        # Get data embedding and train MSE metrics
        self.model = model
        self.z_train, self.rec_train_metrics = model.score(dataset)
        n_components = self.z_train.shape[1]

        # Fit regressions (one per combination of class and latent variable)
        if dataset.latents is not None and not mse_only:
            # Use dummy labels for one class if no class labels are provided or if partition mode is turned off
            if dataset.labels is None or not dataset.partition:
                labels = np.zeros(len(dataset))
            else:
                labels = dataset.labels[:, 0]

            c = np.unique(labels)

            # Check if classes are correctly indexed
            for i in range(len(c)):
                if i != c[i]:
                    raise ValueError('Class labels should be indexed from 0 to no_of_classes - 1.')

            for i in c:
                mask = labels == i
                z_c = self.z_train[mask]
                y_c = dataset.latents[mask]
                self.linear_regressors.append([])

                for j, latent in enumerate(y_c.T):
                    scaler = PolarConverter() if (j in dataset.is_radial and n_components == 2) else StandardScaler()
                    pipeline = Pipeline(steps=[('scaler', scaler),
                                               ('regression', SGDRegressor())])
                    pipeline.fit(z_c, latent)
                    self.linear_regressors[int(i)].append(pipeline)

        # Fit one linear classifier per class of labels
        if dataset.labels is not None and not mse_only:
            for i, classes in enumerate(dataset.labels.T):
                pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regression',
                                                                          LogisticRegression(max_iter=1000))])
                pipeline.fit(self.z_train, classes)
                self.linear_classifiers.append(pipeline)

    def score(self, dataset, is_train=False):
        """Score dataset.

        Args:
            dataset(BaseDataset): Dataset to score.
            is_train(optional, bool): If True, will reuse embeddings and MSE metrics computed during fit.

        Returns:
            (tuple) tuple containing:
                z(ndarray): Data embedding.
                metrics(dict[float]): Dict of metrics.

        """
        metrics = dict()

        if is_train:
            z, rec_metrics = self.z_train, self.rec_train_metrics
        else:
            z, rec_metrics = self.model.score(dataset)

        metrics.update(rec_metrics)

        # Fit regressions (one per combination of class and latent variable)
        if dataset.latents is not None and not self.mse_only:
            r2 = list()

            # Use dummy labels for one class if no class labels are provided or if partition mode is turned off
            if dataset.labels is None or not dataset.partition:
                labels = np.zeros(len(dataset))
            else:
                labels = dataset.labels[:, 0]

            c = np.unique(labels)

            # Check if classes are correctly indexed
            for i in range(len(c)):
                if i != c[i]:
                    raise ValueError('Class labels should be indexed from 0 to no_of_classes - 1.')

            for i in c:
                mask = labels == i
                z_c = z[mask]
                y_c = dataset.latents[mask]

                for j, latent in enumerate(y_c.T):
                    r2.append(self.linear_regressors[int(i)][j].score(z_c, latent))

            metrics.update({'R2': np.mean(r2)})
        else:
            metrics.update({'R2': -1})

        # Fit one linear classifier per class of labels
        if dataset.labels is not None and not self.mse_only:
            acc = list()
            for i, classes in enumerate(dataset.labels.T):
                acc.append(self.linear_classifiers[int(i)].score(z, classes))

            metrics.update({'Acc': np.mean(acc)})
        else:
            metrics.update({'Acc': -1})

        return z, metrics

def get_GRAE_networks(dataA, dataB, emb, n_comp = 2, anchors = [], labelsA = None, labelsB = None):  
    """
    Generate two GRAE networks from dataA and dataB.
    Parameters
    ----------
    dataA : array-like of shape (n_samples_A, n_features)
        Input data for the first dataset.
    dataB : array-like of shape (n_samples_B, n_features)
        Input data for the second dataset.
    emb : array-like of shape (n_samples_A + n_samples_B, emb_dim)
        Embeddings for both datasets concatenated. The first segment corresponds to dataA, 
        and the second segment corresponds to dataB.
    n_comp : int, optional
        Number of components (dimensions) for the GRAE networks, by default 2.
    labelsA : array-like of shape (n_samples_A,), optional
        Class labels for the first dataset. If None, zeros are used.
    labelsB : array-like of shape (n_samples_B,), optional
        Class labels for the second dataset. If None, zeros are used.
    Returns
    -------
    myGraeA : GRAEBase
        Trained GRAE model for dataA.
    myGraeB : GRAEBase
        Trained GRAE model for dataB.
    Notes
    -----
    This function prepares the data for each dataset, sets default labels if not provided, 
    and fits the respective GRAEBase models using the provided embeddings.
    """

    if labelsA is None:
        labelsA = np.zeros(len(dataA))
     
    if labelsB is None:
        labelsB = np.zeros(len(dataB))

    split_A = BaseDataset(x = dataA, y = np.array(labelsA), split_ratio = 0.8, random_state = 42, split = "none")
    myGraeA = anchorGRAE(n_components = n_comp)
    myGraeA.fit(split_A, emb= emb[:len(dataA)], anchors = anchors) 


    #We need to flip the anchors
    anchors = np.array(anchors)
    anchors[:, 0], anchors[:, 1] = anchors[:,1], anchors [:, 0]

    split_B = BaseDataset(x = dataB, y = np.array(labelsB), split_ratio = 0.8, random_state = 42, split = "none")
    myGraeB = anchorGRAE(n_components = n_comp)
    myGraeB.fit(split_B, emb= emb[len(dataA):], anchors = anchors)

    return myGraeA, myGraeB