import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numbers


''' Notations:
- C_in, C_out: number of channels
- H, W, N: dimension of grid (N implies it is square)
- M: size of graph (so the order of N²)
'''


# Utils


def heat_kernel(λ, Φ, t):
    ''' Computes heat kernels at various times, using the D (out of M) eigenvectors provided.
    λ shape D, Φ shape MxD, t shape T (times at which to compute)
    Returns h[t,x,y] (shape TxMxM)
    Complexity: T * M² * D
    '''
    Φ = Φ.t()  # DxM
    outers = Φ[:, None, :] * Φ[:, :, None]  # outers[i,x,y] = Φi[x] Φi[y] (DxMxM)
    f = th.exp(- λ[None, :] * t[:, None])  # TxD
    h = th.tensordot(f, outers, dims=1)  # TxMxM
    return h



# Convolutions

class Conv1x1(nn.Module):
    ''' 1x1 convolutions for either 1D (graph) or 2D (grid) data.
    Number of parameters:  C_in * C_out
    Receptive field:       1 x 1
    Complexity of forward: M * C_out * C_in
    '''
    def __init__(self, in_channels, out_channels, input_ndim):
        super().__init__()
        if input_ndim == 1:
            conv = nn.Conv1d
        elif input_ndim == 2:
            conv = nn.Conv2d
        else:
            raise ValueError('Unknown input number of dimensions: {}'.format(input_ndim))
        self.conv = conv(in_channels, out_channels, kernel_size=1)
        self.input_ndim = input_ndim

    def forward(self, x):
        '''
        Input:  B x C_in x *
        Output: B x C_out x *
        '''
        return self.conv(x).squeeze(-(1 + self.input_ndim))


class ConvGrid(nn.Module):
    ''' Classical convolution on a grid.
    Number of parameters:  C_in * C_out * kernel_size²
    Receptive field:       kernel_size x kernel_size
    Complexity of forward: H * W * C_out * C_in * kernel_size²
    '''
    def __init__(self, **kwargs):
        ''' Forwards its arguments to nn.Conv2d. '''
        super().__init__()
        self.conv = nn.Conv2d(**kwargs)
        self.input_ndim = 2

    def forward(self, x):
        '''
        Input:  B x C_in x H x W
        Output: B x C_out x H+2p-k+1 x W+2p-k+1 (assuming stride 1)
        '''
        return self.conv(x)


class ConvMasked(ConvGrid):
    ''' Classical convolution on a grid, with wall-masking at every step.
    Since it uses the graph structure, it includes half-padding so the input shape remains constant.
    Number of parameters:  C_in * C_out * kernel_size²
    Receptive field:       kernel_size x kernel_size
    Complexity of forward: H * W * C_out * C_in * kernel_size²
    '''
    def __init__(self, G, kernel_size=3, **kwargs):
        '''
        G is the walkability map of shape H x W
        kernel_size is by default 3 to avoid information going over walls
        additional arguments are passed to ConvGrid
        '''
        super().__init__(kernel_size=kernel_size, padding=(kernel_size-1)//2, **kwargs)
        self.G = G

    def forward(self, x):
        '''
        Input:  B x C_in x H x W, assumes x is zero for walls
        Output: B x C_out x H x W, zero for walls (best followed by a ReLU)
        '''
        x = super().forward(x)
        return x * self.G


class ConvSpectral(nn.Module):
    ''' Spectral convolutions on a graph.
    Number of parameters:  C_in * C_out * d
    Receptive field:       infinite
    Complexity of forward: M * C * max(C, D)
    '''
    # TODO: allow to subsample first eigenvectors and use cubic splines to interpolate the missing coefficients
    def __init__(self, λ, Φ, d, in_channels, out_channels):
        ''' Constructs a spectral convolution layer on the supplied graph.
        λ: M, eigenvalues of the laplacian
        Φ: MxM, eigenvectors of the laplacian (in columns)
        '''
        super().__init__()
        self.input_ndim = 1

        # L = vectors diag(values) vectors^T
        self.Φ = Φ[:, :d] # M x D
        self.λ = λ[:d] # D

        self.w = nn.Parameter(th.empty((out_channels, in_channels, d)))
        self.b = nn.Parameter(th.empty((out_channels, d)))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w)
        init.kaiming_uniform_(self.b)

    def forward(self, x):
        '''
        Input:  B x C_in x M
        Output: B x C_out x M
        '''
        x = th.tensordot(x, self.Φ.t(), dims=([2], [1])) # B x C_in x D in spectral basis
        x = th.sum(self.w[None,:,:,:] * x[:,None,:,:], dim=2) + self.b[None,:,:]
        x = th.tensordot(x, self.Φ, dims=([2], [1])) # B x C_out x M in spatial basis
        return x.squeeze(-2)


class ConvPolynomial(nn.Module):
    pass


class ConvHeat(nn.Module):
    ''' Convolutions using learn anisotropic heat kernels.
    This is not implemented yet, as we lack a backward for th.eig() (or a way to do without).
    Besides, the laplacian may not even be positive.
    '''
    def __init__(self, next):
        super(ConvHeat, self).__init__()
        D, M = next.shape
        self.grad = th.eye(M)[None] - th.eye(M)[next]  # DxMxM
        self.div = th.eye(M)[next]  # DxMxM

        self.A = th.empty((D, D))
        self.reset_parameters()

    def reset_paremeters(self):
        self.A = init.kaiming_uniform_(self.A)

    def heat_kernel(self):
        Agrad = th.tensordot(self.A, self.grad, dims=1)  # DxMxM
        divAgrad = th.tensordot(self.div, Agrad, dims=([0, 1], [0, 1]))  # MxM
        L = -divAgrad
        # XXX L is not symmetric and backward is not implemented for eig (but is for symeig)
        pass



# Diffusion


class DiffConv(nn.Module):
    ''' Diffuses information spatially using a convolution, treating channels as batches.
    Number of parameters:  C * num_conv_param (typically k²)
    Receptive field:       same as underlying convolution (but does not mix channels)
    Complexity of forward: M * C * complexity of convolution (typically k²)
    '''
    def __init__(self, channels, conv=ConvGrid, **conv_kwargs):
        '''
        :param conv: the convolution to use
        '''
        super(DiffConv, self).__init__()
        self.conv = conv(in_channels=channels, out_channels=channels, groups=channels, **conv_kwargs)

    def forward(self, x):
        '''
        Input:  * x C x conv_input_shape
        Output: * x C x conv_output_shape
        '''
        return self.conv(x)


class DiffHeat(nn.Module):
    ''' Diffuses info spatially using heat kernels. 
    Number of parameters:  0
    Receptive field:       growing with t
    Complexity of forward: M³ * C
    '''
    def __init__(self, λ, Φ, t):
        super(DiffHeat, self).__init__()
        self.h = heat_kernel(λ, Φ, th.tensor(t).unsqueeze(0)).squeeze(0) # MxM, symmetric

    def forward(self, x):
        '''
        Input:  * x C x M
        Output: * x C x M
        '''
        return th.tensordot(x, self.h, dims=1) # This effectively averages around each node since h.sum(0) = 1


class Smoothing(nn.Module):
    def __init__(self, mask, factor, passes):
        super(Smoothing, self).__init__()
        self.mask = mask
        self.factor = factor
        self.passes = passes

    def smooth_1d(self, input, dim):
        idx = th.arange(input.size(dim) - 2) + 1
        output = th.cat((
            input.select(dim, 0).unsqueeze(dim),
            self.factor * (input.index_select(dim, idx - 1) * self.mask.index_select(dim, idx - 1)
                           + input.index_select(dim, idx + 1) * self.mask.index_select(dim, idx + 1)) + \
            (1 - (self.mask.index_select(dim, idx - 1) + self.mask.index_select(dim, idx + 1)) * self.factor) * \
            input.index_select(dim, idx),
            input.select(dim, -1).unsqueeze(dim),
        ), dim=dim)
        output = self.mask * output + (1 - self.mask) * input
        return output


    def forward(self, input):
        output = input
        for _ in range(self.passes):
            output = self.smooth_1d(self.smooth_1d(output, -1), -2)
        return output


class DiffDistance(nn.Module):
    def __init__(self, dist, diffusion):
        super().__init__()
        self.kernel = th.exp(-dist / diffusion) * (dist >= 0).float()
        self.kernel /= self.kernel.sum((-1, -2), keepdim=True) + 1e-9

    def forward(self, x):
        return th.tensordot(x, self.kernel, dims=2)

# TODO: some sort of pooling? (that wouldn't change input size...), or followed by upsampling...



# Architectures


class Narrowing(nn.Module):
    ''' This architecture successively reduces the grid size.
    Since the input size changes from layer to layer, only non-graph layers can be used.
    Number of parameters:  N * C² * k
    Complexity of forward: N³ * C² * k
    '''
    def __init__(self, input_dim, input_channels, num_channels, kernel_size, activation=nn.ReLU):
        ''' The network is composed of a sequence of blocks.
        Each block is a convolution, followed by an activation.
        The number of blocks is automatically computed so the final spatial size is 
        a 1x1 square (roughly N/k for a NxN grid).
        :param input_dim: the size of the input grid (necessarily square)
        :param input_channels: number of channels in the input
        :param num_channels: number of channels in each layer
        :param kernel_size: kernel size of convolutions
        :param activation: activation used in each block
        '''
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        assert ((self.input_dim - 1) % (self.kernel_size - 1) == 0)
        self.num_conv = (self.input_dim - 1) // (self.kernel_size - 1)
        self.input_channels = input_channels

        self.activation = activation

        layers = []
        for i in range(self.num_conv):
            in_channels = self.num_channels if i > 0 else self.input_channels
            layers.append(ConvGrid(in_channels=in_channels, out_channels=self.num_channels,
                                   kernel_size=self.kernel_size))
            if i < self.num_conv - 1:
                layers.append(self.activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        '''
        Input:  * x input_channels x input_dim x input_dim
        Output: * x num_channels
        '''
        input = obs.view((-1, self.input_channels, self.input_dim, self.input_dim))
        output = self.layers(input)
        return output.view(obs.shape[:-3] + (self.num_channels,))


class Fixed(nn.Module):
    ''' This architecture keeps the input shape fixed.
    Number of parameters:  num_conv * num_channels² * number of parameters of convolution (k² for a grid convolution)
    Complexity of forward: num_conv * num_channels² * N² * complexity of convolution (k² for a grid convolution)
    '''
    def __init__(self, input_channels, num_channels, num_conv, output_channels=1,
                 activation=nn.ReLU, conv=ConvGrid, **conv_kwargs):
        ''' The network is composed of a sequence of blocks, each block being a convolution and a non-linearity.
        :param input_channels: number of channels in the input
        :param num_channels: number of intermediate channels
        :param num_conv: number of blocks in the network
        :param output_channels: number of channels in the output
        :param activation: activation to use
        :param conv: convolution to use
        :param conv_kwargs: additional arguments (beyond in/out channels) to pass to the convolution layers
        '''
        super().__init__()
        self.num_channels = num_channels
        self.num_conv = num_conv
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = conv
        self.conv_kwargs = conv_kwargs.copy()
        if self.conv == ConvGrid:
            self.conv_kwargs['padding'] = (self.conv_kwargs['kernel_size'] - 1) // 2
            # Other convolutions keep a fixed input size.
        self.activation = activation

        layers = []
        for i in range(self.num_conv):
            in_channels = self.num_channels if i > 0 else self.input_channels
            out_channels = self.num_channels if i < self.num_conv - 1 else self.output_channels
            layers.append(self.conv(in_channels=in_channels, out_channels=out_channels, **self.conv_kwargs))
            self.input_ndim = layers[-1].input_ndim # 1D (graph) or 2D (grid) convolution
            if i < self.num_conv - 1:
                layers.append(self.activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        '''
        Input:  * x input_channels x conv_input_shape (1D for graph, or 2D for grid)
        Output: * (x output_channels, squeezed if 1) x conv_input_shape
        '''
        input_dim = obs.shape[-self.input_ndim:]   # last input_ndim dimensions
        batch_dim = obs.shape[:-1-self.input_ndim] # first dimensions
        # obs shape is batch_dim + (input_channels,) + input_dim
        input = obs.view((-1, self.input_channels) + input_dim)
        output = self.layers(input)
        return output.view(batch_dim + ((self.output_channels,) if self.output_channels > 1 else ()) + input_dim)


class Decoupled(nn.Module):
    ''' Alternates 1x1 convolutions (mixes information in channels) and diffusions (mixes information in space).
    Number of parameters:  num_conv * (C² + diffusion_parameters)
    Complexity of forward: num_conv * M * C * (C + diffusion_complexity)
    '''
    def __init__(self, input_channels, num_channels, num_conv, input_ndim, output_channels=1,
                 activation=nn.ReLU, diff=ConvGrid, diff_kwargs=dict()):
        ''' The network is composed of a sequence of blocks, each block being a 1x1 convolution, a diffusion 
        and a non-linearity.
        :param input_channels: number of channels in the input
        :param num_channels: number of intermediate channels
        :param num_conv: number of blocks in the network
        :param input_ndim: 1 or 2, spatial dimensions of input
        :param output_channels: number of channels in the output
        :param activation: activation to use
        :param conv: convolution to use
        :param conv_kwargs: additional arguments (beyond in/out channels) to pass to the convolution layers
        '''
        super(Decoupled, self).__init__()
        self.num_channels = num_channels
        self.num_conv = num_conv
        self.input_channels = input_channels
        self.input_ndim = input_ndim
        self.output_channels = output_channels

        self.diff = diff
        self.diff_kwargs = diff_kwargs.copy()
        self.activation = activation

        layers = []
        for i in range(self.num_conv):
            in_channels = self.num_channels if i > 0 else self.input_channels
            out_channels = self.num_channels if i < self.num_conv - 1 else self.output_channels
            layers.append(Conv1x1(in_channels, out_channels, self.input_ndim))
            if i < self.num_conv - 1:
                layers.append(self.diff(**self.diff_kwargs))
                layers.append(self.activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        '''
        Input:  * x input_channels x diff_input_shape (1D for graph, or 2D for grid)
        Output: * x output_channels x diff_input_shape
        '''
        input_dim = obs.shape[-self.input_ndim:]  # last input_ndim dimensions
        batch_dim = obs.shape[:-1 - self.input_ndim]  # first dimensions
        # obs shape is batch_dim + (input_channels,) + input_dim
        input = obs.view((-1, self.input_channels) + input_dim)
        output = self.layers(input)
        return output.view(batch_dim + ((self.output_channels,) if self.output_channels > 1 else ()) + input_dim)


class Hourglass(nn.Module):
    ''' Hourglass network (encoder-decoder). Uses pooling and upsampling to reduce the number of parameters. '''
    # TODO: problem is pooling implies changing the graph size, or we duplicate the max in the whole bin or something?
    pass



# Misc


class Multinomial(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, features):
        '''
        Input:  * x C
        Output: * long without grad (actions), * float with grad (log of proba(chosen action))
        '''
        # features is * x features_dim
        output = self.softmax(features)  # * x output_dim
        action = output.multinomial(1).squeeze(-1)  # *, long, no grad
        log_prob = th.log(output).gather(-1, action.unsqueeze(-1)).squeeze(-1)  # *, log of proba(chosen action), grad
        return action, log_prob


# Sadly, unpicklable...
class Lambda(nn.Module):
    def __init__(self, λ):
        super().__init__()
        self.λ = λ

    def forward(self, x):
        return self.λ(x)


class Flatten2D(nn.Module):
    def forward(self, x):
        return x.view(x.shape[:-2] + (-1,))


class GatherToGraph(nn.Module):
    def __init__(self, G):
        '''
        :param G: HxW, walkability mask
        '''
        super().__init__()
        self.flatten = Flatten2D()
        self.I = th.nonzero(G.view(-1)).squeeze(1) # M

    def forward(self, x):
        '''
        Input:  * x H x W
        Output: * x M
        '''
        return th.index_select(self.flatten(x), -1, self.I)


class ScatterToGrid(nn.Module):
    def __init__(self, G):
        '''
        :param G: HxW, walkability mask
        '''
        super().__init__()
        self.G = G
        self.I = th.nonzero(G.view(-1)).squeeze(1) # M

    def forward(self, x):
        '''
        Input:  * x M
        Output: * x H x W
        '''
        y = th.zeros(x.shape[:-1] + (self.G.numel(),))
        y.scatter_(dim=-1, index=self.I.expand(x.shape), src=x)
        return y.view(x.shape[:-1] + self.G.shape)
