import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import matplotlib.pyplot as plt
import time

from models import *

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
        obs[0] = obs[-1]
        epsilon *= decay

def dqn(env, network):
    step_batch = 1

    γ = .9
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
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
            plt.imshow(((env.grid != gd.cell_wall).float() + obs[-1, 0, 1]).detach().cpu().numpy())
            plt.subplot(122)
            plt.imshow(network(obs[-1]).detach().cpu().numpy()[0])
            # plt.imshow(Smoothing((env.grid != gd.cell_wall).float(), .1, 5)(obs[-1, :, 0] + obs[-1, :, 1]).detach().cpu().numpy()[0])
            plt.show()
        epoch_start = time.time()


if __name__ == '__main__':
    th.set_default_tensor_type(cuda.FloatTensor)
    import grid as gd
    env = gd.GridEnv(gd.read_grid('grids/7x7.png'), batch=1024 * 16, control='dir')
    dqn(env,
        Decoupled(
            num_channels=4,
            num_conv=2,
            input_channels=2,
            input_ndim=2,
            activation=nn.ReLU,
            diff=Smoothing,
            diff_kwargs=dict(
                mask=(env.grid != gd.cell_wall).float(),
                factor=.2,
                passes=5,
            )
        ),
        # Fixed(
        #     num_channels=16,
        #     num_conv=3,
        #     input_channels=2,
        #     kernel_size=1,
        # )
    )
