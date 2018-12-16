import torch as th
import torch.nn as nn

class Multinomial(nn.Module):
    def __init__(self, features_dim, output_dim):
        super(Multinomial, self).__init__()
        self.linear = nn.Linear(features_dim, output_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, features):
        return th.distributions.Categorical(self.softmax(self.linear(features)))

    def sample(self, features):
        return self.softmax(self.linear(th.zeros(features.shape))).multinomial(1)

    def log_prob(self, features, value):
        return th.log(self.softmax(self.linear(th.zeros(features.shape))))[:, :, 0]

class Policy(nn.Module):
    def __init__(self, feature_network, action_network):
        super(Policy, self).__init__()
        self.feature_network = feature_network
        self.action_network = action_network

    def forward(self, obs):
        features = self.feature_network(obs)
        return self.action_network(features)

    def sample(self, obs):
        features = self.feature_network(obs)
        return self.action_network.sample(features)

    def log_prob(self, obs, value):
        features = self.feature_network(obs)
        return self.action_network.log_prob(features, value)

def collect(env, policy, step_batch):
    obs = th.zeros((step_batch + 1, env.B) + env.obs_shape)
    act = th.zeros((step_batch, env.B), dtype=th.int)
    rew = th.zeros((step_batch, env.B))
    don = th.zeros((step_batch, env.B), dtype=th.int)

    obs[0] = env.reset()

    while True:
        for t in range(step_batch):

            act[t] = policy.sample(obs[t]).view(-1)
            obs[t+1], rew[t], don[t], _ = env.step(act[t])
        yield obs, act, rew, don
        obs[0] = obs[-1]

def compute_return(rew, don, γ):
    with th.no_grad():
        ret = th.zeros(rew.shape)
        T = rew.shape[0]
        for t in range(T):
            γ_l = th.ones((rew.shape[1]))
            for l in range(t, T):
                ret[t] += γ_l * rew[l]
                γ_l *= γ * (1 - don[l]).to(th.float32)
        return ret

def reinforce(env, policy):
    step_batch = 256
    γ = .95
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
    for obs, act, rew, don in collect(env, policy, step_batch):
        ret = compute_return(rew, don, γ)
        optimizer.zero_grad()
        loss = ( - policy.log_prob(obs[:-1], act) * ret).sum()
        loss.backward()
        optimizer.step()
        print('loss={:.2f} rew={:.2f} ret={:.2f}'.format(loss.item(), rew.mean().item(), ret.mean().item()))
        # env.render()

if __name__ == '__main__':
    import gridpt as gt
    env = gt.GridEnv(gt.read_grid('grids/basic.png'), batch=64)
    reinforce(env, Policy(
        nn.Linear(2, 64),
        Multinomial(64, env.D * 2)
    ))
