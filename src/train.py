# Hyperparameters
total_steps = 25e6
num_envs = 64
num_levels = 10000
num_steps = 256
num_epochs = 3
batch_size = 1024#128, uncomment for low VRAM
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
gamma = 0.999

from utils import make_env, Storage, orthogonal_init
# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels, gamma=gamma)
print('Observation space:', env.observation_space.shape)
print('Action space:', env.action_space.n)


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module
    
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x
        
class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels, out_channels = 256,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=64)
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=out_channels)

        self.output_dim = out_channels
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value, maxAction = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def actMax(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value, maxAction = self.forward(x)
      action = maxAction
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    maxAction = logits.argmax(dim=1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value, maxAction


# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels, gamma=gamma)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)



in_channels = 3
nr_features = 256

print('in_channels value ' + str(in_channels))
# Define network
#encoder = Encoder(in_channels, nr_features)
encoder = ImpalaModel(in_channels, nr_features)
print('Test1')
policy = Policy(encoder, nr_features, env.action_space.n )
print("Test2")
policy.cuda()
print('Test3')
# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma
)


# Run training
obs = env.reset()
step = 0
while step < total_steps:
  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value, _ = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      ratio = (new_log_prob - b_log_prob).exp()
      surr1 = ratio * b_advantage
      surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * b_advantage

      # Clipped policy objective
      pi_loss = - torch.min(surr1, surr2).mean()
      # Clipped value function objective
      value_loss = (new_value - b_returns ).pow(2).mean()

      # Entropy loss
      entropy_loss = torch.mean(torch.exp(new_log_prob) * new_log_prob)

      # Backpropagate losses
      loss = pi_loss + value_coef * value_loss + entropy_coef * entropy_loss
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')

print('Completed training!')
torch.save(policy.state_dict, 'checkpoint.pt')

import imageio

# Make evaluation environment
eval_env = make_env(num_envs, start_level=0, num_levels=num_levels, gamma=gamma)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

  # Use policy
  action, log_prob, value = policy.actMax(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid.mp4', frames, fps=25)