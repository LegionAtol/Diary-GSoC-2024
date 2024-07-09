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

### With Gymnasium and Stable Baselines3
As seen [Gymnasium](https://gymnasium.farama.org) is useful for using environments or creating custom environments.  
This integrates well with [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) which provides various RL algorithms in PyTorch.  

We can recreate the example above in a faster way, this time using the Advantage Actor-Critic (A2C) algorithm with a policy called "MlpPolicy" (based on a multi-layer perceptron neural network)  

install the following packages:
```python
pip install gymnasium
pip install stable_baselines3
pip install pygame 
pip install opencv-python   
```

The program to run:
```python
import gymnasium as gym
from stable_baselines3 import A2C

# Creating the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# create a A2C model using MlpPolicy policy
model = A2C("MlpPolicy", env, verbose=1)
# model training to improve policy
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
# loop to test the trained model
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human") # renderizzo l'ambiente
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset() 
```
### Custom Environment - GridWorld Example
In this section we see how to create a custom environment with Gymnasium and we will use a Stablebaseline3 policy.  
The environment in question is a 5x5 grid and an agent that must move in this 2 dim grid to reach a target state.  
The allowed actions are up, down, right and left and you cannot leave the grid. The state to be reached is box (3,3)  

To [define a custom environment in Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) the main things to do are:  
Extend gym.Env, define an action space (to indicate which actions the agent can perform), define an observation space (the values ​​that the environment can return ), define the step method (will contain the program logic, the interaction between agent and environment) and define the reset method (called when an episode is concluded).  

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

class SimpleGrid2DEnv(gym.Env):
    def __init__(self):
        super(SimpleGrid2DEnv, self).__init__()
        self.grid_size = 5
        self.goal_position = (3, 3)
        self.action_space = spaces.Discrete(4)  #0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        self.state = np.array([0, 0], dtype=np.int32)
        
        self.rewards = []  # All episodes rewards
        self.cumulative_reward = 0  # Rewards for the current episode
        self.max_steps = 50  # Maximum steps per episode
        self.current_step = 0  # Current step in the episode
        self.episodes_length = [] # Keep the number of steps used in each episode

    def reset(self, seed=None, options=None):
        if hasattr(self, 'cumulative_reward') and self.cumulative_reward != 0:  # Skip the first reset
            self.rewards.append(self.cumulative_reward)
        self.cumulative_reward = 0  # Reset the cumulative reward for the new episode
        self.episodes_length.append(self.current_step)  # save the number of step used in each episode
        self.current_step = 0  # Reset the step counter
        self.state = np.array([0, 0], dtype=np.int32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1:
            self.state[0] += 1
        elif action == 2 and self.state[1] > 0:
            self.state[1] -= 1
        elif action == 3 and self.state[1] < self.grid_size - 1:
            self.state[1] += 1

        reward = 1 if tuple(self.state) == self.goal_position else -0.1
        self.cumulative_reward += reward
        terminated = tuple(self.state) == self.goal_position
        truncated = self.current_step >= self.max_steps
        #truncated = False
        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == tuple(self.state):
                    grid[i, j] = 'A'
                elif (i, j) == self.goal_position:
                    grid[i, j] = 'G'
                else:
                    grid[i, j] = '.'
        print("\n".join(["".join(row) for row in grid]))
        print()

if __name__ == "__main__":
    # Create the environment
    env = SimpleGrid2DEnv()

    # Check that the environment is valid
    check_env(env, warn=True)

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=15000)

    # Save the model
    #model.save("ppo_simple_grid_2d")

    # Caricare il modello (opzionale)
    #model = PPO.load("ppo_simple_grid_2d")

    # Print the latest cumulative reward during training
    if len(env.rewards) > 0:
        print(f"Last cumulative reward during training is {env.rewards[-1]}")

    test_rewards = []

    for test in range(5):
        # Test the trained agent
        obs, _ = env.reset()
        env.render()
        for _ in range(20):
            action, _states = model.predict(obs)
            print(f"Test {test + 1}, Action: {action}")  # Debug print
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Test {test + 1}, State: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")  # Debug print
            env.render()
            if terminated or truncated:
                break
        
        # Print the cumulative test episode reward
        print(f"Cumulative reward of test {test + 1} is {env.cumulative_reward}")
        test_rewards.append(env.cumulative_reward)

    # Print all cumulative test rewards
    print("Cumulative rewards for all tests:", test_rewards)

    # Number of steps used in each episode
    #print(f"Number of steps used in each epsiode:\n{env.episodes_length}")

    # Create the reward chart during training
    plt.plot(env.rewards, label='Cumulative Reward')
    
    # Highlight episodes with the highest reward
    max_reward = max(env.rewards)
    max_indices = [i for i, reward in enumerate(env.rewards) if reward == max_reward]
    plt.scatter(max_indices, [env.rewards[i] for i in max_indices], color='red', label='Max Reward', zorder=5)

    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward per Episode during Training')
    plt.legend()
    plt.show()
```
Note that the **reward function** is defined so as not only to be positive (for example reward = +1 if the target is reached, otherwise reward =0) but is also defined with a penalty term based on the number of steps it takes to reach the target state.  
Designing an effective reward function for the problem is crucial in RL.  
Running the code with : total_timesteps=10000  
I obtained a cumulative reward graph like the following:  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/f47e5963-6f6e-4e29-8391-9401d6b06a57" alt="image" width="900"/>

After training, the model is saved and then reloaded to perform 5 tests by starting the agent in the same initial position.  
I got:  
Cumulative rewards for all tests: [0.4, 0.10000000000000009, -0.8000000000000005, -0.09999999999999987, -2.0000000000000004]  
As you can see, the PPO agent still acts stochastically even if it starts from the same initial state.  
We can say that yes, he has learned, but not yet very well...  
However, we can see that the longer the training, the less stochastic it acts.  
With a train of total_timesteps=15000 I got:  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/e824e627-71be-4454-9319-287c62ba1fe1" alt="image" width="900"/>

From the tests I got:
Cumulative rewards for all tests: [0.5, 0.5, 0.3, 0.5, 0.5]
Practically the probability of doing "exploration" is much lower, it is almost always selecting the best path found so far (which is not a local trap, but actually the best one).

Those seen so far were without considering the **truncated signal** (just put in the step method, truncated=False always), this was possible because we have episodic tasks, i.e. the episode ends sooner or later because the agent will reach the target state.  
All episodes ended with a **terminated signal**.  
Using the truncated signal for example if a maximum number of steps is exceeded (for example max_steps = 50) during the episode, things change slightly (keeping total_timesteps=15000) as can also be seen from the graph:  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/e165508e-3a1c-4629-a626-5c4da15c773d" alt="image" width="900"/>

And the cumulative rewards for all tests: [0.5, 0.30000000000000004, -0.5000000000000002, 0.5, 0.5]  

We can think of the Policy that the PPO agent learns as a "weighted average" in the sense that it considers all the episodes it has seen, but will give more value to those ending with higher rewards.  
From what has been observed, it does not seem to make much sense to apply "restore best weight" or "Early stopping" techniques.  
Another confirmation that the agent is learning is that as the training progresses, the episodes become less and less long, until the minimum number of steps to reach the target state is reached.  

Note that episodes that end in a truncated signal are not discarded, but are still used differently to update the policy.  
It might be useful to keep track of how many episodes are truncated and how many end correctly, so that the latter are the majority.  
You can see the comparison of terimanted and truncated in the [documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
