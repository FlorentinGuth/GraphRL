import torch as th
import matplotlib.pyplot as plt
import queue
from concurrent.futures import ProcessPoolExecutor

cell_empty = 0x808080
cell_wall = 0x000000
cell_start = 0x000080


def _compute_dist_from(args):
    grid, dirs, init = args
    dists = th.full(grid.shape, -1)
    q = queue.Queue(grid.shape[0] * grid.shape[1])
    q.put((0, init))
    while not q.empty():
        d, u = q.get()
        if dists[u] < 0 and grid[u] != cell_wall:
            dists[u] = d
            for dir in dirs:
                v = tuple(th.tensor(u) + dir)
                if grid[v] != cell_wall:
                    q.put((d + 1, v))
    return init, dists

class GridEnv:
    def __init__(self, grid, batch=1, timeout=-1, seed=0, control='direction'):
        """
        grid is a D-dimensional array, 1 for walkable, 0 for non-walkable. It is assumed to have borders of 0s
        batch (B) is the number of parallel trajectories to simulate

        a pos is BxD array
        an action is an integer, index in the directions array

        control is either 'direction' or 'node'
        """
        self.grid = grid
        self.init = th.nonzero(self.grid == cell_start)[0]
        self.D = self.grid.ndimension()
        self.B = batch
        self.pos = self.init.repeat((self.B, 1)) # BxD

        self.seed = seed
        self.dust_prob = self.generate_dust_prob()
        self.dust = th.zeros((self.B,) + self.grid.shape)

        self.time = th.zeros(self.B, dtype=th.long)
        self.dirs = th.zeros((2*self.D, self.D), dtype=th.long) # AxD
        actions = th.arange(2*self.D)
        self.dirs[actions, actions//2] = 2*(actions % 2) - 1
        self.obs_shape = (3,) + self.grid.size() # grid, position, dust
        self.timeout = timeout
        self.control = control
        if control == 'node':
            self.dists = self.compute_dists()

    def compute_dists(self):
        with ProcessPoolExecutor(max_workers=20) as executor:
            dists = th.full(self.grid.shape + self.grid.shape, -1)
            size = self.grid.size(0)
            grid, dirs = self.grid.cpu(), self.dirs.cpu()
            for init, dist in executor.map(_compute_dist_from, [(grid, dirs, (i // size, i % size)) for i in range(size ** 2)]):
                dists[init] = dist
        return dists

    def generate_dust_prob(self):
        th.manual_seed(self.seed)

        # Parameters of the generation
        num_seeds = 5
        diameter = sum(self.grid.shape)  # approximation of longest shortest path
        max_proba = 1. / (2 * diameter)
        min_proba = max_proba / 3
        num_iter = 10

        walkability = self.grid != cell_wall  # HxW, byte
        sum_neighbours = lambda g: g[0:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, 0:-2] + g[1:-1, 2:]
        num_neighbours = sum_neighbours(walkability)  # H-2xW-2, byte
        w = walkability[1:-1, 1:-1].to(th.float32)  # H-2xW-2, float

        # Seeds: random corners/corridors of the grid
        seeds = (w * (num_neighbours <= 2).to(th.float32)).nonzero()  # Nx2
        perm = th.randperm(seeds.shape[0])[:min(num_seeds, seeds.shape[0])]
        seeds = seeds[perm]

        # Iterate diffusion from the seeds
        num_averages = (num_neighbours + 1).to(th.float32)  # H-2xW-2, float
        dust_prob = th.zeros(self.grid.shape)
        dust_prob[1:-1, 1:-1][tuple(seeds.t())] = min_proba + \
                                                  (max_proba - min_proba) * th.rand(seeds.size(0))
        for _ in range(num_iter):
            dust_prob[1:-1, 1:-1] = w * (sum_neighbours(dust_prob) + dust_prob[1:-1, 1:-1]) / num_averages

        return dust_prob

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
        if self.control == 'node':
            if len(a.shape) < 2:
                a = th.stack((a // self.grid.shape[0], a % self.grid.shape[0]), dim=-1)
            dist = self.dists[tuple(a.t().unsqueeze(-1)) +
                              tuple((self.pos[:, None, :] + self.dirs[None, :, :]).permute(2, 0, 1))].clone()
            dist[dist < 0] = dist.max() + 1
            a = dist.argmin(1)
        new_pos = self.pos + self.dirs[a]
        walkable = self.grid[tuple(new_pos.t())] != cell_wall
        reward = walkable.to(th.float32) - 1
        self.pos[walkable] = new_pos[walkable]

        idx = (self.dust == 0) & (self.dust_prob > 0)
        self.dust[idx] = self.dust_prob.expand_as(self.dust)[idx].bernoulli()
        reward += self.dust[(th.arange(self.B),) + tuple(self.pos.t())]
        self.dust[(th.arange(self.B),) + tuple(self.pos.t())] = 0

        reward -= .01 * self.dust.sum((1, 2))

        self.time += 1
        done = th.zeros(self.B, dtype=th.uint8)
        if self.timeout > 0:
            done = self.time > self.timeout
        self.time[done] = 0
        self.pos[done] = self.init

        return self.observation(), reward, done, None

    def render(self, figure=None, title=None):
        if self.D != 2:
            raise ValueError('Impossible to render non-2D')

        if figure is not None:
            figure.clear()
        if title is not None:
            plt.title(title)
        render_grid(self.grid)
        plt.imshow(self.dust_prob, alpha=0.5)
        y, x = self.pos.t()
        plt.scatter(x, y, marker='x')
        y, x = th.nonzero(self.dust[0]).t()
        plt.scatter(x, y, marker='o')
        if figure is not None:
            figure.canvas.flush_events()
            figure.canvas.draw_idle()
            figure.canvas.flush_events()

def read_grid(file):
    grid = plt.imread(file, int).astype(int)
    return th.from_numpy((grid[:, :, 0] << 16) + (grid[:, :, 1] << 8) + grid[:, :, 2]).to(th.zeros(0).device)

def render_grid(grid):
    plt.imshow(th.stack((
        (grid >> 16) & 0xFF,
        (grid >> 8 ) & 0xFF,
        (grid >> 0 ) & 0xFF,
    ), dim=-1))
