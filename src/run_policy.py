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

    policy = th.load('policy.pth').to(th.device('cpu'))
    env = grid.GridEnv(grid.read_grid('grids/basic.png'))
    env.render()
    obs = env.reset()
    mean_reward = 0
    while True:
        obs, rew, _, _ = env.step(policy(obs)[0])
        mean_reward = mean_reward * (1 - alpha) + alpha * rew.item()
        env.render(fig, 'Reward: cur={:.2f} mean={:.2f}'.format(rew.item(), mean_reward))
        time.sleep(.01)
