import torch.optim as optim
import torch.cuda as cuda
import psutil
import time

from models import *


class Policy(nn.Module):
    ''' Combines networks
    observation --feature_network-----action_network--> action, log_prob
                                   |
                                   ----value_netowrk--> value
    '''
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
    horizon = 256
    γ = .98
    value_factor = 3e-4
    optimizer = optim.Adam(policy.parameters(), lr=3e-3)
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
    env = gd.GridEnv(gd.read_grid('grids/25x25.png'), batch=64 * 1, control='node')
    reinforce(env, Policy(
        nn.Sequential(),
        nn.Sequential(
            Fixed(
                input_channels=2,
                num_channels=16,
                kernel_size=3,
                num_conv=3,
                ),
            Flatten2D(),
            Multinomial(),
        ),
        nn.Sequential(
            Narrowing(
                input_dim=env.obs_shape[1],
                num_channels=16,
                kernel_size=3,
                input_channels=2,
                ),
            nn.Linear(16, 1),
        ),
    ))
