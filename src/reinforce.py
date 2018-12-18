import torch as th
import torch.nn as nn

class Multinomial(nn.Module):
    def __init__(self, features_dim, output_dim):
        super(Multinomial, self).__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(features_dim, output_dim)
        self.softmax = nn.Softmax(-1)

    def sample(self, features):
        return self.softmax(self.linear(features)).multinomial(1)

    def log_prob(self, features, value):
        return th.log(self.softmax(self.linear(features))).gather(-1, value.unsqueeze(-1).type(th.LongTensor)).squeeze(-1)

class Policy(nn.Module):
    def __init__(self, feature_network, action_network, value_network):
        super(Policy, self).__init__()
        self.feature_network = feature_network
        self.action_network = action_network
        self.value_network = value_network

    def sample(self, obs):
        features = self.feature_network(obs)
        return self.action_network.sample(features)

    def log_prob(self, obs, value):
        features = self.feature_network(obs)
        return self.action_network.log_prob(features, value)

    def value(self, obs):
        features = self.feature_network(obs)
        return self.value_network(features)

def collect(env, policy, step_batch):
    with th.no_grad():
        obs = th.zeros((step_batch + 1, env.B) + env.obs_shape)
        act = th.zeros((step_batch, env.B), dtype=th.long)
        rew = th.zeros((step_batch, env.B))
        don = th.zeros((step_batch, env.B), dtype=th.uint8)

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
    γ = .9
    value_factor = 1e-3
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-2)
    for obs, act, rew, don in collect(env, policy, step_batch):
        ret = compute_return(rew, don, γ)
        val = policy.value(obs[:-1]).squeeze(-1)
        with th.no_grad():
            adv = ret - val
        loss = ( - policy.log_prob(obs[:-1], act) * adv + value_factor * (ret - val) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss={:.2f} rew={:.2f} ret={:.2f}, val={:.2f}'.format(
            loss.item(), rew.mean().item(), ret.mean().item(), val.mean().item()))

if __name__ == '__main__':
    import grid as gd
    env = gd.GridEnv(gd.read_grid('grids/basic.png'), batch=1024, timeout=128)
    reinforce(env, Policy(
        nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
        ),
        Multinomial(64, env.D * 2),
        nn.Linear(64, 1),
    ))
