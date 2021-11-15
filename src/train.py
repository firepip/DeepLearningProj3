import sys
# Hyperparameters
total_steps = 25e6
if len(sys.argv) > 1:
    print("Total steps: " + sys.argv[1])
    total_steps = int(sys.argv[1])
num_envs = 64
num_levels = 10000
if len(sys.argv) > 2:
  print("Num levels: " + sys.argv[2])
  num_levels = int(sys.argv[2])

num_steps = 256
num_epochs = 3
batch_size = 128#, uncomment for low VRAM
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
gamma = 0.999
env_name='starpilot'
if len(sys.argv) > 3:
  print("Game: " + sys.argv[3])
  env_name = sys.argv[3]

model = 1
if len(sys.argv) > 4:
  print("Model: " + sys.argv[4])
  model = int(sys.argv[4])


from utils import make_env, Storage, orthogonal_init
# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels, gamma=gamma, env_name = env_name)
#print('Observation space:', env.observation_space.shape)
#print('Action space:', env.action_space.n)


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init


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
env = make_env(num_envs, num_levels=num_levels, gamma=gamma, env_name = env_name)
#print('Observation space:', env.observation_space)
#print('Action space:', env.action_space.n)



in_channels = 3
nr_features = 256

#print('in_channels value ' + str(in_channels))
# Define network
#encoder = Encoder(in_channels, nr_features)

from Impala import ImpalaModel

if model == 1:
  encoder = ImpalaModel(in_channels, nr_features)
#TODO add more models

policy = Policy(encoder, nr_features, env.action_space.n )
policy.cuda()

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
#obs = augment(obs)
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
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels, gamma=gamma, env_name = env_name)
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