import torch as th
import matplotlib.pyplot as plt

cell_empty = 0x808080
cell_wall = 0x000000
cell_start = 0x000080
cell_intermediate = 0x008000
cell_end = 0x800000

class GridEnv:
    def __init__(self, grid, batch=1):
        """
        grid is a D-dimensional array, 1 for walkable, 0 for non-walkable. It is assumed to have borders of 0s
        batch (B) is the number of parallel trajectories to simulate

        a pos is BxD array
        an action is an integer, index in the directions array
        """
        self.grid = grid
        self.size = th.tensor(self.grid.size())
        self.init = th.nonzero(self.grid == cell_start)[0]
        self.D = self.grid.ndimension()
        self.B = batch
        self.pos = self.init.repeat((self.B, 1)) # BxD
        self.found_intermediate = th.zeros(self.B, dtype=th.uint8)
        self.time = th.zeros(self.B, dtype=th.long)
        self.dirs = th.zeros((2*self.D, self.D), dtype=th.long) # AxD
        actions = th.arange(2*self.D)
        self.dirs[actions, actions//2] = 2*(actions % 2) - 1

    def step(self, a):
        """
        a is B array of actions
        returns observation of new state, reward, done, debug
        """
        new_pos = self.pos + self.dirs[a]
        walkable = self.grid[tuple(new_pos.t())] != cell_wall
        self.pos[walkable] = new_pos[walkable]
        # | and & are elementwise-bitwise 'or' and 'and'
        self.found_intermediate = self.found_intermediate | (self.grid[tuple(self.pos.t())] == cell_intermediate)
        done = self.found_intermediate & (self.grid[tuple(self.pos.t())] == cell_end)
        self.time += 1
        finish_times = self.time[done]
        self.time[done] = 0
        if done.any():
            self.pos[done] = self.init
        return self.pos, walkable.to(th.float32) - 1, done, finish_times

    def render(self):
        if self.D != 2:
            raise ValueError('Impossible to render non-2D')
        plt.figure()
        render_grid(self.grid)
        y, x = self.pos.T
        plt.scatter(x, y, marker='x')
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
