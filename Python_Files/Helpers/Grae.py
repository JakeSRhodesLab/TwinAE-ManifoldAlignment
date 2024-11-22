"""Base class for datasets.

All data will be saved to a data/processed/dataset_name folder."""
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

FIT_DEFAULT = .85  # Default train/test split ratio
SEED = 42  # Default seed for splitting

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
        """Fit model to data.

        Args:
            x(BaseDataset): Dataset to fit.

        """
        raise NotImplementedError()

    def fit_transform(self, x):
        """Fit model and transform data.

        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of X.

        Args:
            x(BaseDataset): Dataset to fit and transform.

        Returns:
            ndarray: Embedding of x.

        """
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        """Transform data.

        If model is a dimensionality reduction method, such as an Autoencoder, this should return the embedding of x.

        Args:
            X(BaseDataset): Dataset to transform.
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

    def fit_plot(self, x_train, x_test=None, cmap='jet', s=15, title=None):
        """Fit x_train and show a 2D scatter plot of x_train (and possibly x_test).

        If x_test is provided, x_train points will be smaller and grayscale and x_test points will be colored.

        Args:
            x_train(BaseDataset): Data to fit and plot.
            x_test(BaseDatasset): Data to plot. Set to None to only plot x_train.
            cmap(str): Matplotlib colormap.
            s(float): Scatter plot marker size.
            title(str): Figure title. Set to None for no title.

        """
        self.plot(x_train, x_test, cmap, s, title, fit=True)

    def plot(self, x_train, x_test=None, cmap='jet', s=15, title=None, fit=False, figsize = (10, 7)):
        """Plot x_train (and possibly x_test) and show a 2D scatter plot of x_train (and possibly x_test).

        If x_test is provided, x_train points will be smaller and grayscale and x_test points will be colored.
        Will log figure to comet if Experiment object is provided. Otherwise, plt.show() is called.

        Args:
            x_train(BaseDataset): Data to fit and plot.
            x_test(BaseDatasset): Data to plot. Set to None to only plot x_train.
            cmap(str): Matplotlib colormap.
            s(float): Scatter plot marker size.
            title(str): Figure title. Set to None for no title.
            fit(bool): Whether model should be trained on x_train.

        """
        if self.comet_exp is not None:
            # If comet_exp is set, use different backend to avoid display errors on clusters
            matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
        import matplotlib.pyplot as plt

        if not fit:
            z_train = self.transform(x_train)
        else:
            z_train = self.fit_transform(x_train)

        y_train = x_train.targets.numpy()

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
            y_test = x_test.targets.numpy()
            plt.scatter(*z_train.T, c='grey', s=s / 10, alpha=.2)
            plt.scatter(*z_test.T, c=y_test, cmap=cmap, s=s)

        if self.comet_exp is not None:
            self.comet_exp.log_figure(figure=plt, figure_name=title)
            plt.clf()
        else:
            plt.show()

    def reconstruct(self, x):
        """Transform and inverse x.

        Args:
            x(BaseDataset): Data to transform and reconstruct.

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
            x(BaseDataset): Dataset to sample from.
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
        x, _ = x.numpy()

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

    def view_surface_rec(self, x, n_max=1000, random_state=42, title=None, dataset_name=None):
        """View 3D original surface and reconstruction.

        Only call this method on 3D surface datasets. x is expected to be 2D.
        Will show figure or log it to Comet if self.comet_exp was set.

        Args:
            x(BaseDataset): Dataset to sample from.
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
        x, y = x.numpy()

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