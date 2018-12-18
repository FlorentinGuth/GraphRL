import torch as th
import matplotlib.pyplot as plt

cell_empty = 0x808080
cell_wall = 0x000000
cell_start = 0x000080
cell_dust = 0x800000

class GridEnv:
    def __init__(self, grid, batch=1, timeout=-1):
        """
        grid is a D-dimensional array, 1 for walkable, 0 for non-walkable. It is assumed to have borders of 0s
        batch (B) is the number of parallel trajectories to simulate

        a pos is BxD array
        an action is an integer, index in the directions array
        """
        self.grid = grid
        self.init = th.nonzero(self.grid == cell_start)[0]
        self.D = self.grid.ndimension()
        self.B = batch
        self.pos = self.init.repeat((self.B, 1)) # BxD
        self.dust_prob = (.1 * (grid == cell_dust).to(th.float32)).repeat((self.B, 1, 1)) # BxHxW
        self.dust = th.zeros((self.B,) + self.grid.shape)

        self.time = th.zeros(self.B, dtype=th.long)
        self.dirs = th.zeros((2*self.D, self.D), dtype=th.long) # AxD
        actions = th.arange(2*self.D)
        self.dirs[actions, actions//2] = 2*(actions % 2) - 1
        self.obs_shape = (3,) + self.grid.size() # grid, position, dust
        self.timeout = timeout

    def observation(self):
        obs = th.zeros((self.B,) + self.obs_shape)
        obs[:, 0] = self.grid != cell_wall
        obs[(th.arange(self.B), 1) + tuple(self.pos.t())] = 1
        obs[:, 2] = self.dust
        return obs

    def reset(self):
        self.pos[:] = self.init
        self.dust[:] = 0
        return self.observation()

    def step(self, a):
        """
        a is B array of actions
        returns observation of new state, reward, done, debug
        """
        new_pos = self.pos + self.dirs[a]
        walkable = self.grid[tuple(new_pos.t())] != cell_wall
        reward = walkable.to(th.float32) - 1
        self.pos[walkable] = new_pos[walkable]

        idx = (self.dust == 0) & (self.dust_prob > 0)
        self.dust[idx] = self.dust_prob[idx].bernoulli()
        self.dust[(th.arange(self.B),) + tuple(self.pos.t())] = 0

        reward -= self.dust.sum((1, 2))

        self.time += 1
        done = th.zeros(self.B, dtype=th.uint8)
        if self.timeout > 0:
            done = self.time > self.timeout
        self.time[done] = 0
        self.pos[done] = self.init

        return self.observation(), reward, done, None

    def render(self):
        if self.D != 2:
            raise ValueError('Impossible to render non-2D')
        plt.figure()
        render_grid(self.grid)
        y, x = self.pos.t()
        plt.scatter(x, y, marker='x')
        y, x = th.nonzero(self.dust[0]).t()
        plt.scatter(x, y, marker='o')
        plt.show()

def read_grid(file):
    grid = plt.imread(file, int).astype(int)
    return th.from_numpy((grid[:, :, 0] << 16) + (grid[:, :, 1] << 8) + grid[:, :, 2])

def render_grid(grid):
    plt.imshow(th.stack((
        (grid >> 16) & 0xFF,
        (grid >> 8 ) & 0xFF,
        (grid >> 0 ) & 0xFF,
    ), dim=-1))
