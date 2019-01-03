import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import matplotlib.pyplot as plt
from reinforce import *
import time

class ConvGridFixed(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, num_conv):
        super(ConvGridFixed, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.num_conv = num_conv

        layers = []
        for i in range(self.num_conv):
            in_channels = self.num_channels if i > 0 else 2
            layers.append(nn.ZeroPad2d((self.kernel_size - 1) // 2)),
            layers.append(nn.Conv2d(in_channels, self.num_channels, self.kernel_size))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(self.num_channels, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        input = obs.view((-1, 2, self.input_dim, self.input_dim))
        output = self.layers(input)
        return output.squeeze(-3)

def collect(env, network, step_batch, γ):
    obs = [None] * (step_batch + 0 + 1)
    act = [None] * (step_batch + 0)
    val = [None] * (step_batch + 0)
    rew = [None] * (step_batch + 0)
    don = [None] * (step_batch + 0)
    ret = [None] * (step_batch + 0)

    epsilon = 1
    decay = .999

    obs[0] = env.reset()

    values = network(obs[0])
    current_pos = env.pos.data
    action_values = values[(th.arange(env.B).unsqueeze(-1).expand((env.B, env.D * 2)),) + tuple(
        current_pos.t()[:, :, None] + env.dirs.t()[:, None, :])]

    while True:
        for t in range(step_batch):
            act[t] = action_values.argmax(-1)
            act[t] = act[t].data
            bernoulli = th.full(act[t].shape, epsilon).bernoulli()
            act[t][bernoulli > 0] = th.randint(env.dirs.shape[0], (int(bernoulli.sum().item()),))
            val[t] = action_values[th.arange(act[t].shape[0]), act[t]]
            obs[t + 1], rew[t], don[t], _ = env.step(act[t])

            values = network(obs[t + 1])
            current_pos = env.pos.data
            action_values = values[(th.arange(env.B).unsqueeze(-1).expand((env.B, env.D * 2)),) + tuple(
                current_pos.t()[:, :, None] + env.dirs.t()[:, None, :])]

            ret[t] = rew[t] + γ * action_values.max(-1)[0].data

        def to_tensor(x):
            return x if hasattr(x, 'shape') else th.stack(x[:step_batch], 0)
        yield tuple(map(to_tensor, (obs, act, val, rew, don, ret)))
        epsilon *= decay

def dqn(env, network):
    step_batch = 1

    γ = .9
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    epoch = 0
    epoch_start = time.time()
    for obs, act, val, rew, don, ret in collect(env, network, step_batch, γ):
        loss = ((ret - val) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.save(network, 'network.pth')
        epoch += 1
        elapsed = time.time() - epoch_start
        print('epoch={}, rew={:.2f}, ret={:.2f}, val={:.2f}, speed={}'.format(
            epoch, rew.mean().item(), ret.mean().item(), val.mean().item(), int(step_batch * env.B / elapsed)))
        if epoch % 5000 == 0:
            plt.subplot(121)
            plt.imshow(env.grid.detach().cpu().numpy())
            plt.subplot(122)
            plt.imshow(network(obs[0]).detach().cpu().numpy()[0])
            plt.show()
        epoch_start = time.time()


if __name__ == '__main__':
    th.set_default_tensor_type(cuda.FloatTensor)
    import grid as gd
    env = gd.GridEnv(gd.read_grid('grids/9x9.png'), batch=1024 * 16, control='dir')
    dqn(env,
        nn.Sequential(
            ConvGridFixed(env.obs_shape[1], 16, 3, 5),
        ),
    )
