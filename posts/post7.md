# REINFORCE
Here we talk about the REINFORCE technique which is also mentioned in some papers of my GSoC proposal.  
REINFORCE is a **policy gradient** method. The idea is to learn the parametric policy $\pi_\theta (s,a)$ directly instead of a Value function.  

The objective is to maximize the expected reward:
$J(\theta) = E_{\tau\sim\pi_\theta} [\sum_t r(s_t,a_t)]$
where $\tau$ represents a trajectory (episode) obtained by following $\pi_\theta (s,a)$.  

If we can find the parameters $\theta$ that maximize J, we have solved the problem.  

The parameters $\theta$ can be updated using the gradient ascent technique:  
$\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$  
$\alpha$ is the learning rate.  

Using the **Policy Gradient Theorem**, the gradient of $J$ with respect to $\theta$ is calculated as:  
$\nabla J(\theta) = E_{\tau\sim\pi_\theta} [R(\tau) \sum_t \nabla_\theta \log\pi_\theta(a_t|s_t)]$  
Where $R(\tau)$ is the total return of the episode and at each step t of the episode, it calculates the gradient of the log-probability of the action taken with respect to the policy parameters.  
At the end of the episode, we can multiply it by $R(\tau)$ and then update $\theta$ using the formula above.  

To achieve a more accurate update, only the future rewards influenced by the current action can be considered.  
Instead of $R(\tau)$ calculated at the end of the episode, we use the return $G_t$, resulting in:  
$\nabla J(\theta) = E_{\tau\sim\pi_\theta} [\sum_t (G_t \nabla_\theta \log\pi_\theta(a_t|s_t))]$  
This is the typical formula of the REINFORCE algorithm.  

Sometimes, a **baseline** is also added to reduce the variance:  
$\nabla J(\theta) = E_{\tau\sim\pi_\theta} [\sum_t ((G_t -b(s_t))\nabla_\theta \log\pi_\theta(a_t|s_t))]$  
A common choice for the baseline $b$ is the average of the discounted returns of the episode.  

The idea is to maximize $\sum_t (G_t \log\pi_\theta(a_t|s_t))$, which means minimizing the loss = - $\sum_t (G_t \log\pi_\theta(a_t|s_t))$  
and the parameters can be updated in the direction opposite to the gradient of the loss:  
$\theta \leftarrow \theta - \alpha\nabla_\theta \text{loss}$  

For **discrete action spaces**, the policy is often represented as $\pi_\theta(a|s) = P(a|s,\theta)$, the probability of taking an action in a certain state, given certain parameters $\theta$.  
The policy could be a neural network with a softmax output $P(a|s,\theta) =$ softmax.  

For **continuous action spaces**, the policy is often represented as a Gaussian distribution: $\pi_\theta (a,s)= \mathcal{N}(\mu_{\theta}(s),\sigma_{\theta}(s))$.  
The policy can be modeled by a neural network that produces the mean and standard deviation of the distribution.  

## Example
Here we implement the REINFORCE algorithm with the ['Cart Pole'](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment defined in Gymnasium.  
The goal is to keep a pole upright by performing only two actions at each time step: moving the cart left or right.  
The rewards are defined simply: +1 for each step with the pole balanced ("balanced" means the pole is within a certain narrow angle from vertical).

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/d21bccb0-4861-402e-98b6-c420e461ac12" alt="image" width="300"/>

The policy used is a neural network that takes a state as input (a NumPy array with 4 elements, shape (4,)), has a hidden layer with 32 fully connected neurons, and an output layer with 2 neurons (for the two actions) to which a softmax is applied to get the probabilities of the two possible actions.  

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import numpy as np


# Policy definition
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32) # input layer & hidden layer
        self.fc2 = nn.Linear(32, action_dim) # hidden layer & output layer

    def forward(self, x):   # called via __call__ of nn.Module
        x = F.relu(self.fc1(x)) # x is the state
        action_probs = F.softmax(self.fc2(x), dim=-1)   # probs. of two actions (0 = go left, 1 = go right)
        return action_probs # tensor of dim (1, num_actions)

# Function to choose an action based on probabilities
def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0) # convert to a torch tensor
    action_probs = policy(state)    # calls the forward method
    action = torch.multinomial(action_probs, 1).item() # torch.multinomial(probs=[0.2 , 0.8], num_samples=1) -> will more likely give the index 1
    return action, action_probs

# Update the policy
def update_policy(policy, optimizer, rewards, log_probs, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]: # for convenience I start from the end
        R = r + gamma * R   # immediate r + discounted future rewards
        discounted_rewards.insert(0, R)
    
    discounted_rewards = torch.tensor(discounted_rewards)
    # to reduce the variance a baseline can be used
    baseline = discounted_rewards.mean()
    discounted_rewards = discounted_rewards - baseline
    # normalize for convenience
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    
    loss = 0
    # zip is for pairing each log-probability with the corresponding discounted return (returns an iterator of tuples).
    for log_prob, reward in zip(log_probs, discounted_rewards):
        loss += -log_prob * reward

    optimizer.zero_grad()
    loss.backward() # calculate gradients
    optimizer.step() # Gradient descent rule, update parameters

# Environment
env = gym.make('CartPole-v1')
policy = Policy(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

num_episodes = 1000
reward_history = []

# main loop
for episode in range(num_episodes):
    state, _ = env.reset()  # With each episode, the environment is reset to get the initial state.
    log_probs = []
    rewards = []
    
    # t for the time step in the epsiode (max length 200)
    for t in range(1, 200):
        action, action_probs = select_action(policy, state)
        log_prob = torch.log(action_probs.squeeze(0)[action])
        log_probs.append(log_prob)
        
        state, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        
        # there are defined cases, for which the episode ends earlier (for example if the pole falls down)
        if done:
            break

    update_policy(policy, optimizer, rewards, log_probs) # after each episodes
    reward_history.append(sum(rewards))
    
    # Every 100 episodes
    if episode % 100 == 0:
        print(f'Episode {episode}, total reward: {sum(rewards)}')

env.close()

# Plot
def plot_rewards(reward_history, window=100):
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history, label='Total Reward per Episode')
    if len(reward_history) >= window:
        moving_avg = [np.mean(reward_history[i-window:i]) for i in range(window, len(reward_history))]
        plt.plot(range(window, len(reward_history)), moving_avg, label='Moving Average (100 episodes)', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.show()

plot_rewards(reward_history)

```

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/dcea8927-9b83-42db-bc7f-03053806e193" alt="image" width="500"/>  

As seen from the graph, the agent seems to be learning, with the rewards it gets in various episodes increasing steadily.  
Sometimes there are noticeable spikes downwards, and there are several factors that could cause this behavior:  


- The REINFORCE algorithm can still be subject to high variance.
- The policy selects actions in a stochastic manner, meaning it might choose non-optimal actions (to ensure some exploration).
- The number of neurons, the learning rate, and in general, the hyperparameters can be changed.
- More sophisticated policies could be used, such as the soft ϵ-greedy policy.
