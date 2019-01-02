if __name__ == '__main__':
    import torch as th
    import grid
    from reinforce import *
    import matplotlib.pyplot as plt
    import time

    alpha = .01

    plt.ion()
    fig = plt.figure()
    plt.show()

    policy = th.load('policy.pth', map_location='cpu')
    env = grid.GridEnv(grid.read_grid('grids/25x25.png'), control='node')
    env.render()
    obs = env.reset()
    mean_reward = 0
    step = 0
    while True:
        step += 1
        act = policy(obs)[0]
        # act = (obs[0, 2] + obs[0, 0]).view(-1).argmax().unsqueeze(0)
        obs, rew, _, _ = env.step(act)
        mean_reward = mean_reward * (1 - alpha) + alpha * rew.item()
        env.render(fig, 'Reward: cur={:.2f} mean={:.2f}, step={}'.format(rew.item(), mean_reward, step))
        time.sleep(.02)
