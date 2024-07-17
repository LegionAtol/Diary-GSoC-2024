# Qubit example
## State transfer - Hadamard rotation, 1 step

To understand how to make the algorithm take just **one step** to reach the final state, I decided to take a step back and **manually simulate a Hadamard rotation** by writing a simple program using Qutip functions.  
As can be seen from the code I have defined a drift Hamiltonian for the qubit with sigma z operator, which defines the rotation around the z axis.  
 The control Hamiltonian is defined with the sigma x operator, this is then multiplied by a constant alpha value.  

```python
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define constants and parameters
omega = 2 * np.pi * 3.9  # GHz
alpha = 12  # Control parameter
times = np.linspace(0, 0.1)  # Time frame for evolution (one step)

# Pauli operators
sx = sigmax()
sy = sigmay()
sz = sigmaz()

# Hamiltonians
H_drift = (omega / 2) * sz  # Drift Hamiltonian
H_control = sx  # Control Hamiltonian

# Total Hamiltonian
H = H_drift + alpha * H_control

# Initial state (up state)
psi0 = basis(2, 0)

# calculate the evolution of the system
result = mesolve(H, psi0, times)

# Track intermediate ket vectors
kets = result.states

# To visualize the trajectory of the intermediate kets on the Bloch sphere
bloch = Bloch()
for ket in kets:
    bloch.add_states(ket)
bloch.show()
```  

Given the dynamics of the system, the objective now is to find the value of alpha that allows in a single step, i.e. in as little time as possible, starting from state |0> to perform a Hadamard transformation and therefore reach state |+> with a very high fidelity.  
After a few attempts by manually changing the alpha value, I arrive at alpha equal to around 11 or 12 and graphically you can observe the evolution of the system with mesolve at various moments of time:  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/86741a5b-5789-4407-859e-05f5369fdb80" alt="image" width="300"/>  

Now let's go back to the RL problem  

In the code present in the previous post, I used many parameters as they were initially used in the paper cited at the beginning of this post.  
I decided to simplify a bit by using C1 = 1, slightly simplify the penalty term in the reward function, set u_max = 1 knowing that the range of actions will be limited in Box and above all set a range for the action space (defined by the Gymnasium Box) from -15 to +15 knowing that the optimal action is approximately 12.  
By doing so, however, you have a very **large actions space** and the algorithm with each training manages to reach the final state in approximately 7, 8 or 9 steps with almost total safety (in the sense that even by doing various trainings, towards the end it takes example with 7 minimum steps for many episodes in a row.  
Perhaps it can be thought of as if they were local minima due to the fact that the algorithm was unable to explore the entire state-action space well and increasing training time does not appear to further reduce steps per episode).  

Gymnasium and RL Algorithms seem to work best by normalizing the **Box** function range for actions **between -1 and 1**.  
Then I used a constant **u_max** to multiply by the action (in our case we know that the optimal value is close to 12, so I use u_max = 13 to give it a slightly wider range) so as to have a larger "virtual" interval [-13, 13]  
With a train of total_timesteps=200000 and max step = 100, it reaches the end of the train with still variation in the episodes as visible:  
Number of steps per episode:  
[40,…. 5, 10, 21, 22, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 10, 6, 1, 8, 3, 15, 1, 23, 1, 3, 5]  
From this I can observe two things:   
First of all, we need to lengthen the train time as towards the end the policy is still "exploring" a lot, it has not found a minimum number of steps.  
Secondly, we can also see that he has already found episodes where the policy only took one step, which is a good thing.  
In this case with total_timesteps=200000 or more we see that towards the end of the train it almost always reaches a single step for many episodes in a row.  

Furthermore, the more I narrow the range of action space, the better.  
One way would be to **shift the range**:  
The interval defined in the Box class is always [-1,1] but then I translate the selected action into [9, 13] as follows:  
alpha = ((action + 1) / 2 * (13 - 9)) + 9  
Obviously I choose the interval [9, 13] if I know that the value of the action will be in that interval.  
By doing so, the algorithm managed to reach a fidelity > 99% with one step with a total_timesteps less than 100000.  
In fact, the best action found was: 11.018799441  

The code is as follows  

```python
import gymnasium as gym
from gymnasium import spaces
import qutip as qu
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class GymQubitEnv(gym.Env):
    def __init__(self):
        super(GymQubitEnv, self).__init__()
        
        self.dim = 2  # dimension of Hilbert space
        self.u_max = 13 
        self.w = 2 * np.pi * 3.9  # (GHz)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Observation space, |v> have 2 real and 2 imaginary numbers -> 4

        # time for mesolve()
        self.time_end = 0.1

        # threshold for fidelity to consider the target state reached
        self.fidelity_threshold = 0.99 

        self.current_step_in_the_episode = 0
        self.max_steps = 100    # max episode length (max number of steps for one episode)

        # Reward parameters
        self.C1 = 1    
        self.step_penalty = 1  # step penalty parameter

        # Target state after applying Hadamard gate to |0>
        self.target_state = (qu.gates.hadamard_transform() * qu.basis(self.dim, 0)).unit()  # Hadamard applied to |0>
        self.state = None   # actual state

        # Hamiltonians
        self.H_0 = self.w / 2 * qu.sigmaz()
        self.H_1 = qu.sigmax()

        #for debugging
        self.episode_reward = 0 
        self.rewards = []   # contains the cumulative reward of each episode
        self.fidelities = []    # contains the final fidelity of each episode
        self.highest_fidelity = 0  # track the highest fidelity achieved
        self.highest_fidelity_episode = 0  # track the episode where highest fidelity is achieved
        self.episode_actions = []  # Track actions for the current episode
        self.actions = []  # Contains the actions taken in each episode
        self.num_of_terminated = 0  # number of episodes terminated
        self.num_of_truncated = 0   # number of truncated episodes
        self.episode_steps = [] # number of steps for each episodes
        
        self.seed = None

    def step(self, action):
        action = action[0]  # action is an array -> extract it's value with [0]
        #alpha = action * self.u_max # the action is limited between -u_max , +u_max. 
        alpha = ((action + 1) / 2 * (13 - 9)) + 9   # Scale action from [-1, 1] to [9, 13]
        H = self.H_0 + alpha * self.H_1

        result = qu.mesolve(H, self.state, [0, self.time_end])
        self.state = result.states[-1] # result.states returns a list of state vectors (kets), is a a Qobj object. let's take the last one.

        fidelity = qu.fidelity(self.state, self.target_state)
        reward = self.C1 * fidelity - self.step_penalty
        self.current_step_in_the_episode += 1
        terminated = fidelity >= self.fidelity_threshold    # if the goal is reached
        truncated = self.current_step_in_the_episode >= self.max_steps  # if the episode ended without reaching the goal
        #truncated=False

        reward = float(reward.item())  # Ensure the reward is a float

        # for debugging
        #print(f"Step {self.current_step_in_the_episode}, Fidelity: {fidelity}")
        self.episode_reward += reward
        self.episode_actions.append(action)
        if terminated or truncated:
            self.fidelities.append(fidelity) # keep the final fidelity
            if fidelity > self.highest_fidelity:
                self.highest_fidelity = fidelity  # update highest fidelity
                self.highest_fidelity_episode = len(self.rewards) + 1  # update the episode number (since rewards are appended after reset)
            self.rewards.append(self.episode_reward) # keep the episode rewards
            self.episode_reward = 0  # Reset the episode reward
            self.episode_steps.append(self.current_step_in_the_episode) # Keep the number of steps used for this episode
            self.current_step_in_the_episode = 0  # Reset the step counter
            self.actions.append(self.episode_actions.copy()) # Append actions of the episode to the actions list
            self.episode_actions = []  # Reset the actions for the new episode
        if terminated:
            self.num_of_terminated += 1
        elif truncated:
            self.num_of_truncated += 1

        observation = self._get_obs()
        
        return observation, reward, bool(terminated), bool(truncated), {"state": self.state}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.state = self.create_init_state(noise=False)

        return self._get_obs(), {}

    # if state=(p q)' with p = a + i*b and q = c + i*d -> return [a, b, c, d]
    def _get_obs(self):
        rho = self.state.full().flatten() # to have state vector as NumPy array and flatten into one dimensional array.[a+i*b c+i*d]
        obs = np.concatenate((np.real(rho), np.imag(rho)))
        return obs.astype(np.float32) # Gymnasium expects the observation to be of type float32

    def create_init_state(self, noise=False, random=False):
        if random:
            # Randomly choose |0> or |1> with equal probability
            if np.random.rand() > 0.5:
                init_state = qu.basis(self.dim, 1)  # |1>
            else:
                init_state = qu.basis(self.dim, 0)  # |0>
        else:
            if noise:
                # Initial slight variations of |0>
                perturbation = 0.1 * (np.random.rand(self.dim) - 0.5) + 0.1j * (np.random.rand(self.dim) - 0.5) # to get something like: [0.03208387-0.01834318j 0.0498474 -0.0339512j ]
                perturbation_qobj = qu.Qobj(perturbation, dims=[[self.dim], [1]])
                init_state = qu.basis(self.dim, 0) + perturbation_qobj
                init_state = init_state.unit()  # to ensure unitary norm
            else:
                init_state = qu.basis(self.dim, 0)  # |0>
        return init_state


if __name__ == '__main__':
    env = GymQubitEnv()

    # Check if the environment follows Gym API
    check_env(env, warn=True)

    # Create the model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100000) #200000
 
    # For debugging
    print("\n Summary of the trining:")
    for i, (r, f) in enumerate(zip(env.rewards, env.fidelities), start=1):
        #print(f"Rewards for episode {i}: {r}")
        print(f"Fidelity for episode {i}: {f}")
        if i % 50 == 0:
            avg_reward = np.mean(env.rewards[i-50:i])
            avg_fidelity = np.mean(env.fidelities[i-50:i])
            print(f"Episode {i}, Avg reward of last 50 episodes: {avg_reward}")
            print(f"Episode {i}, Avg fidelity of last 50 episodes: {avg_fidelity}\n")

    print(f"Highest fidelity achieved during training: {env.highest_fidelity}")
    print(f"Highest fidelity was achieved in episode: {env.highest_fidelity_episode}")
    print(f"Number of: Terminated episodes {env.num_of_terminated}, Truncated episodes {env.num_of_truncated}")
    print(f"Number of steps used in each episode {env.episode_steps}")
    
    # Plot actions of some episodes
    # the action chosen at each step remains constant during the evolution of the system with mesolve, 
    # therefore in the plot I represent them constant 
    num_episodes = len(env.actions)
    indices = [9, 19, num_episodes - 11, num_episodes - 1]  # 10th, 20th, (final-10)th, final episodes
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    for i, idx in enumerate(indices):
        steps = np.arange(len(env.actions[idx]))  # Create an array of step indices
        actions = env.actions[idx]  # Extract action values from the array  
        # Plot each action as a constant value over its interval
        axs[i].step(steps, actions, where='post')
        axs[i].set_title(f'Episode {idx + 1}')
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel('Action')
        print(f"The actions of episode{num_episodes - 1}\n {env.actions[num_episodes - 1]}")   #to see the numerical values ​​of the shares
    plt.tight_layout()
    #print(f"The actions of episode{num_episodes - 1}\n {env.actions[num_episodes - 1]}")   #to see the numerical values ​​of the shares


    # Test the model
    num_tests = 10 # Number of tests to perform
    max_steps = 100 # max number of steps in eatch test
    figures = []  # List to store the figures
    target_state = (qu.gates.hadamard_transform() * qu.basis(env.dim, 0)).unit()  # Hadamard applied to |0>

    for test in range(num_tests):
        print(f"\nTest {test + 1}")
        obs, _ = env.reset()  # Reset the environment to get a random initial state
        initial_state = env.state  # Save the initial state
        #all_intermediate_states = []  # if you want to view all intermediate states
        for _ in range(max_steps):
            action, _states = model.predict(obs, deterministic=False)  # Get action from the model
            obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
            #all_intermediate_states.append(info["state"])  # Collect all final states from the steps
            if _ == max_steps-1:
                final_state = info["state"]  # Get the final state from the environment
                print(f"Test episode not ended! final Fidelity achived: {qu.fidelity(final_state, target_state)}")
            if terminated or truncated:  # Check if the episode has ended
                # Compute fidelity between final state and target state
                final_state = info["state"]  # Get the final state from the environment
                fidelity = qu.fidelity(final_state, target_state)
                print("Final Fidelity:", fidelity)
                # Visualize on the Bloch sphere
                b = qu.Bloch()
                b.add_states(initial_state)
                #b.add_states(all_intermediate_states)  # Add all states to the Bloch sphere
                b.add_states(final_state)   # comment this out if you use b.add_states(all_intermediate_states)
                b.add_states(env.target_state) 
                fig = plt.figure()  # Create a new figure
                b.fig = fig  # Assign the figure to the Bloch sphere
                b.render()  # Render the Bloch sphere
                figures.append(fig)  # Store the figure in the list
                break  # Exit the loop if the episode has ended         
    # Show all figures together
    plt.show()
```  

If you try to set the system's mesolve step evolution time to half, the algorithm will no longer be able to reach the final state with just one step, but by carrying out the training as expected it will find the minimum steps to be 2.  
However, the value of the actions for the two steps is not necessarily the same.  
If you want to try to have the same value for the two actions (so as to think of the two smaller steps as a single longer step), you could probably modify the reward function so as to penalize the actions the more different they are.  
