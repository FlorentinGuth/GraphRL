import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import matplotlib.pyplot as plt
import time
import argparse

from models import *

def collect(env, network, step_batch, γ, epsilon=1, decay=.999, max_step=-1):
    obs = [None] * (step_batch + 0 + 1)
    act = [None] * (step_batch + 0)
    val = [None] * (step_batch + 0)
    rew = [None] * (step_batch + 0)
    don = [None] * (step_batch + 0)
    ret = [None] * (step_batch + 0)

    obs[0] = env.reset()

    values = network(obs[0])
    current_pos = env.pos.data
    action_values = values[(th.arange(env.B).unsqueeze(-1).expand((env.B, env.D * 2)),) + tuple(
        current_pos.t()[:, :, None] + env.dirs.t()[:, None, :])]

    step = 0
    while max_step < 0 or step < max_step:
        for t in range(step_batch):
            act[t] = ((action_values - action_values.mean(-1, keepdim=True)) /
                      (action_values.std(-1, keepdim=True) + 1e-3) * 5).softmax(-1).multinomial(1).squeeze(-1)
            act[t] = act[t].data
            if epsilon > 0:
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
        step += step_batch

def dqn(env, network, title):
    step_batch = 1

    γ = .98
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    epoch = 0
    epoch_start = time.time()

    n_epochs = 50000
    rewards = th.zeros(n_epochs)
    for obs, act, val, rew, don, ret in collect(env, network, step_batch, γ, decay=.9998):
        loss = ((ret - val) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        th.save(network, 'storage/network-{}.pth'.format(title))
        epoch += 1
        elapsed = time.time() - epoch_start
        print('epoch={}, rew={:.2f}, ret={:.2f}, val={:.2f}, speed={}'.format(
            epoch, rew.mean().item(), ret.mean().item(), val.mean().item(), int(step_batch * env.B / elapsed)))
        rewards[(epoch-1) % n_epochs] = rew.mean()
        if epoch % n_epochs == 0:
            th.save(rewards, 'storage/rewards-{}.pth'.format(title))
            plt.plot(range(n_epochs), rewards.detach().cpu().numpy())
            plt.savefig('storage/plot-{}'.format(title))
            plt.subplot(121)
            plt.imshow(((env.grid != gd.cell_wall).float() + obs[-1, 0, 1]).detach().cpu().numpy())
            plt.subplot(122)
            plt.imshow(network(obs[-1]).detach().cpu().numpy()[0])
            plt.show()
        epoch_start = time.time()

def enjoy(env, network):
    step = 0
    alpha = .01

    plt.ion()
    fig = plt.figure()
    plt.show()

    mean_reward = 0

    for obs, act, val, rew, don, ret in collect(env, network, 1, .98, epsilon=0):
        step += 1
        mean_reward = mean_reward * (1 - alpha) + alpha * rew.item()
        env.render(fig, 'Reward: cur={:.2f} mean={:.2f}, step={}'.format(rew.item(), mean_reward, step),
                   network(obs[-1]).detach().cpu().numpy()[0])
        time.sleep(.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enjoy', action='store_const', const=True, default=False)
    parser.add_argument('--title', action='store', default='untitled')
    parser.add_argument('--size', action='store', default=7)
    args = parser.parse_args()

    if args.enjoy:
        import grid as gd
        env = gd.GridEnv(gd.read_grid('grids/{N}x{N}.png'.format(N=args.size)), batch=1, control='dir')
        network = th.load('storage/network-{}.pth'.format(args.title), map_location='cpu')
        # network.layers[1] = Smoothing(mask=(env.grid != gd.cell_wall).float(),
        #             factor=.2,
        #             passes=10,)
        enjoy(env, network)
    else:
        th.set_default_tensor_type(cuda.FloatTensor)
        import grid as gd
        env = gd.GridEnv(gd.read_grid('grids/{N}x{N}.png'.format(N=args.size)), batch=1024 * 4, control='dir')
        dqn(env,
            nn.Sequential(
                # GatherToGraph(env.walkability),
                Decoupled(
                    num_channels=8,
                    num_conv=2,
                    input_channels=3,
                    input_ndim=2,
                    activation=nn.Tanh,
                    diff=DiffDistance,
                    diff_kwargs=dict(
                        dist=env.dists.float(), diffusion=1,
                    )
                ),
                # ConvSpectral(env.λ, env.Φ, 10, 3, 8),
                # nn.Tanh(),
                # ScatterToGrid(env.walkability),
                # Conv1x1(8, 1, 2),
                # Fixed(3, 8, 2, activation=nn.Tanh, kernel_size=3)
            ),
            title=args.title,
        )
