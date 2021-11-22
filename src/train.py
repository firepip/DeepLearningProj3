import sys
from random import randrange

# Hyperparameters
from src.Impala_5_layers import FiveBlocksImpala
from src.Impala_leaky_relu import LeakyImpalaModel

total_steps = 25e6
if len(sys.argv) > 1:
    total_steps = int(int(sys.argv[1]) * 1e6)
    print("Total steps: " + str(total_steps))
num_envs = 64
num_levels = 10
if len(sys.argv) > 2:
    print("Num levels: " + sys.argv[2])
    num_levels = int(sys.argv[2])

eval_frequency = 1e6
num_steps = 256
num_epochs = 3
batch_size = 128#1024  # , uncomment for low VRAM
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
gamma = 0.99
env_name = 'starpilot'
if len(sys.argv) > 3:
    print("Game: " + sys.argv[3])
    env_name = sys.argv[3]

model = 1
if len(sys.argv) > 4:
    print("Model: " + sys.argv[4])
    model = int(sys.argv[4])

augmentationMode = 5
if len(sys.argv) > 5:
    print("Augmentation mode: " + sys.argv[5])
    augmentationMode = int(sys.argv[5])

nr_features = 256
if len(sys.argv) > 6:
    print("nr_features: " + sys.argv[6])
    nr_features = sys.argv[6]
useHoldoutAugmentation = False
augmentationModeValidation = 0
if len(sys.argv) > 7:
    print("Augmentation mode validation: " + sys.argv[7])
    augmentationModeValidation = int(sys.argv[7])
    useHoldoutAugmentation = True

randomAugmentation = False

if augmentationMode > 4:
    randomAugmentation = True

parameter_str = "P"
for i in range(1, len(sys.argv)):
    parameter_str += "_" + sys.argv[i]

print(parameter_str)

from utils import make_env, Storage, orthogonal_init

env = make_env(num_envs, num_levels=num_levels, gamma=gamma, env_name=env_name)

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
env = make_env(num_envs, num_levels=num_levels, gamma=gamma, env_name=env_name)

in_channels = 3

from Impala import ImpalaModel

if model == 1:
    encoder = ImpalaModel(in_channels, nr_features)

if model == 2:
    encoder = LeakyImpalaModel(in_channels, nr_features)

if model == 3:
    encoder = FiveBlocksImpala(in_channels, nr_features)

if model == 4:
    encoder = FiveBlocksImpala(in_channels, nr_features)

policy = Policy(encoder, nr_features, env.action_space.n)
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

def resetAugmentationMode():
    if (randomAugmentation):
        setRandomAugmentationMode(num_envs)
    else:
        setAugmentationMode(augmentationMode, num_envs)

from augment import setHoldoutAgumentation, setRandomAugmentationMode, setAugmentationMode, augment

if useHoldoutAugmentation:
    setHoldoutAgumentation(augmentationModeValidation)


def evaluate(step, testEnv, testEnvAugmentationMode = 0):
    if testEnv:
        setAugmentationMode(testEnvAugmentationMode, num_envs)
    # Make evaluation environment
    startlvl = 0
    if testEnv:
        startlvl = num_levels
    eval_env = make_env(num_envs, start_level=startlvl, num_levels=num_levels, gamma=gamma, env_name=env_name)
    obs = eval_env.reset()

    total_reward = []

    # Evaluate policy
    policy.eval()
    for _ in range(2048):
        # Use policy
        action, log_prob, value = policy.actMax(augment(obs))

        # Take step in environment
        obs, reward, done, info = eval_env.step(action)
        total_reward.append(torch.Tensor(reward))

    # Calculate average return
    total_reward = torch.stack(total_reward).sum(0).mean(0)
    if testEnv:
        resetAugmentationMode()
    return total_reward


resetAugmentationMode()


# Run training
obs = augment(env.reset())
# obs = augment(obs)
step = 0
lastEval = step
while step < total_steps:
    # Use policy to collect data for num_steps steps
    policy.eval()
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(obs)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)
        # Update augmentation mode if we have random augmentations
        if (randomAugmentation and randrange(3) == 0):
            setRandomAugmentationMode(num_envs)

        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)

        # Update current observation
        obs = augment(next_obs)

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
            new_dist, new_value, _ = policy(augment(b_obs))
            new_log_prob = new_dist.log_prob(b_action)

            ratio = (new_log_prob - b_log_prob).exp()
            surr1 = ratio * b_advantage
            surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * b_advantage

            # Clipped policy objective
            pi_loss = - torch.min(surr1, surr2).mean()
            # Clipped value function objective
            value_loss = (new_value - b_returns).pow(2).mean()

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
    if ((step - lastEval) >= eval_frequency or step >= total_steps):
        trainScore = evaluate(step, False)
        testScore = evaluate(step, True, augmentationModeValidation)
        print(f'Step: {step}\t({trainScore},{testScore})')
        lastEval = step

print('Completed training!')
torch.save(policy.state_dict, 'checkpoint.pt')

import imageio

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels, gamma=gamma, env_name=env_name)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(2048):
    # Use policy
    action, log_prob, value = policy.actMax(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)

    total_reward.append(torch.Tensor(reward))

    # Render environment and store
    frame = (torch.Tensor(eval_env.render(mode='rgb_array')) * 255.).byte()
    frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)
print('Generating video' + parameter_str + 'total_reward' + total_reward + '_vid.mp4')
# Save frames as video
frames = torch.stack(frames)
imageio.mimsave(parameter_str + 'total_reward' + total_reward + '_vid.mp4', frames, fps=25)
