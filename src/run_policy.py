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
    env = grid.GridEnv(grid.read_grid('grids/7x7.png'))
    env.render()
    obs = env.reset()
    mean_reward = 0
    step = 0
    while True:
        step += 1
        obs, rew, _, _ = env.step(policy(obs)[0])
        mean_reward = mean_reward * (1 - alpha) + alpha * rew.item()
        env.render(fig, 'Reward: cur={:.2f} mean={:.2f}, step={}'.format(rew.item(), mean_reward, step))
        time.sleep(.2)
