import torch as th
import torch.nn as nn


def heat_kernel(L, t):
    ''' 
    L shape NxN, t shape T (times at which to compute)
    Returns h[t,x,y] (shape TxNxN)
    '''
    λ, V = th.symeig(L, eigenvectors=True) # N, NxN
    V = V.T # vectors in lines instead of columns
    outers = V[:,None,:] * V[:,:,None] # outers[i,x,y] = Φi[x] Φi[y] (NxNxN)
    f = th.exp(- λ[None,:] * t[:,None]) # TxN
    h = th.tensordot(f, outers, dims=1) # TxNxN
    return h

class ConvGrid(nn.Module):
    ''' Classical convolution on a grid. '''
    def __init__(self, **kwargs):
        self.conv = nn.Conv2d(**kwargs)

    def forward(self, x):
        '''
        Input: B x C_in x H x W
        Output: B x C_out x H-k+1 x W-k+1
        '''
        return self.conv(x)

class ConvGridMasked(ConvGrid):
    def __init__(self, G, in_channels, out_channels, kernel_size=3):
        '''
        G is the walkability map of shape H x W
        '''
        super(self, ConvGridMasked).__init__(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.G = G

    def forward(self, x):
        '''
        Input: B x C_in x H x W, assumes x is zero for walls
        Output: B x C_out x H x W, zero for walls (best followed by a ReLU)
        '''
        x = super(self, ConvGridMasked).forward(x)
        return x * self.G

# COnv1x1 then diffusion then repeat
# hourglass network

class ConvSpectral(nn.Module):
    ''' Spectral convolutions on a graph. '''
    def __init__(self, L, d, in_channels, out_channels):
        ''' Constructs a spectral convolution layer on the supplied graph.
        L: N x N, the laplacian of the graph (assumed symmetric)
        d: number of eigenvectors to consider (between 0 and N)
        '''
        e, V = th.symeig(L, eigenvectors=True)
        # L = vectors diag(values) vectors^T
        self.V = V # N x N
        nn.Linear()

    def forward(self, x):
        '''
        Input: shape B x C_in x N
        Output: shape B x C_out x N
        '''
        x = th.matmul(self.V.t(), x) # B x C_in x N in spectral basis
        # C_in x N -> C_out x N
        # mat-vector product in each N
        # resp[f] = α stim[f]
        x = None
        x = th.matmul(self.V, x) # B x C_out x N in spatial basis


class Multinomial(nn.Module):
    def __init__(self):
        super(Multinomial, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, features):
        # features is * x features_dim
        output = self.softmax(features)  # * x output_dim
        action = output.multinomial(1).squeeze(-1)  # *, long, no grad
        log_prob = th.log(output).gather(-1, action.unsqueeze(-1)).squeeze(-1)  # *, log of proba(chosen action), grad
        return action, log_prob


class ConvGrid(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size):
        super(ConvGrid, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        assert ((self.input_dim - 1) % (self.kernel_size - 1) == 0)
        self.num_conv = (self.input_dim - 1) // (self.kernel_size - 1)

        layers = []
        for i in range(self.num_conv):
            in_channels = self.num_channels if i > 0 else 3
            layers.append(nn.Conv2d(in_channels, self.num_channels, self.kernel_size))
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        input = obs.view((-1, 3, self.input_dim, self.input_dim))
        output = self.layers(input)
        return output.view(obs.shape[:-3] + (self.num_channels,))


class ConvGridFixed(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, num_conv):
        super(ConvGridFixed, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.num_conv = num_conv

        layers = []
        for i in range(self.num_conv):
            in_channels = self.num_channels if i > 0 else 3
            layers.append(nn.ZeroPad2d(1)),
            layers.append(nn.Conv2d(in_channels, self.num_channels, self.kernel_size))
            layers.append(nn.Tanh())
        layers.append(nn.Conv2d(self.num_channels, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        input = obs.view((-1, 3, self.input_dim, self.input_dim))
        output = self.layers(input)
        return output.view(obs.shape[:-3] + (-1,))
