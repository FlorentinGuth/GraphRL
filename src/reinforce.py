import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import psutil
import time

class Multinomial(nn.Module):
    def __init__(self, features_dim, output_dim):
        super(Multinomial, self).__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(features_dim, output_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, features):
        # features is * x features_dim
        output = self.softmax(self.linear(features)) # * x output_dim
        action = output.multinomial(1).squeeze(-1) # *, long, no grad
        log_prob = th.log(output).gather(-1, action.unsqueeze(-1)).squeeze(-1) # *, log of proba(chosen action), grad
        return action, log_prob


class ConvGrid(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size):
        super(ConvGrid, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        assert((self.input_dim - 1) % (self.kernel_size - 1) == 0)
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


class Policy(nn.Module):
    def __init__(self, feature_network, action_network, value_network):
        super(Policy, self).__init__()
        self.feature_network = feature_network
        self.action_network = action_network
        self.value_network = value_network

    def forward(self, obs):
        # obs is * x env.obs_shape
        features = self.feature_network(obs)
        action, log_prob = self.action_network(features)
        value = self.value_network(features).squeeze(-1) # *
        return action, log_prob, value # *, *, *


def collect(env, policy, step_batch, horizon, γ):
    obs = [None] * (step_batch + horizon + 1)
    act = [None] * (step_batch + horizon)
    lgp = [None] * (step_batch + horizon)
    val = [None] * (step_batch + horizon)
    rew = [None] * (step_batch + horizon)
    don = [None] * (step_batch + horizon)

    obs[0] = env.reset()

    for t in range(horizon):
        act[t], lgp[t], val[t] = policy(obs[t])
        obs[t + 1], rew[t], don[t], _ = env.step(act[t])
    while True:
        for t in range(horizon, horizon + step_batch):
            act[t], lgp[t], val[t] = policy(obs[t])
            obs[t+1], rew[t], don[t], _ = env.step(act[t])
        ret = compute_return(rew, don, γ, step_batch, horizon)
        def to_tensor(x):
            return x if hasattr(x, 'shape') else th.stack(x[:step_batch], 0)
        yield list(map(to_tensor, (obs, act, lgp, val, rew, don, ret)))
        for x in (obs, act, lgp, val, rew, don):
            x[:horizon] = x[-horizon:]



def compute_return(rew, don, γ, step_batch, horizon):
    ret = th.zeros((step_batch,) + rew[0].shape)
    for t in range(step_batch):
        γ_l = th.ones((rew[0].shape[0]))
        for l in range(horizon):
            ret[t] += γ_l * rew[t + l]
            γ_l *= γ * (1 - don[t + l]).to(th.float32)
    return ret


def reinforce(env, policy):
    step_batch = 32
    horizon = 64
    γ = .9
    value_factor = 3e-5
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    epoch = 0
    epoch_start = time.time()
    for obs, act, lgp, val, rew, don, ret in collect(env, policy, step_batch, horizon, γ):
        adv = ret - val.data
        loss = (-lgp * adv + value_factor * (ret - val) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.save(policy, 'policy.pth')
        epoch += 1
        elapsed = time.time() - epoch_start
        print('epoch={}, rew={:.2f}, ret={:.2f}, val={:.2f}, speed={}'.format(
            epoch, rew.mean().item(), ret.mean().item(), val.mean().item(), int(step_batch * env.B / elapsed)))
        epoch_start = time.time()


if __name__ == '__main__':
    th.set_default_tensor_type(cuda.FloatTensor)
    import grid as gd
    env = gd.GridEnv(gd.read_grid('grids/7x7.png'), batch=1024 * 6)
    reinforce(env, Policy(
        ConvGrid(env.grid.shape[0], 16, 3),
        Multinomial(16, env.D * 2),
        nn.Linear(16, 1),
    ))
