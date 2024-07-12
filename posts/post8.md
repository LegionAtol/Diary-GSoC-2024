# Qubit Example
In one of the papers ("[A differentiable programming method for quantum control](https://iopscience.iop.org/article/10.1088/2632-2153/ab9802/pdf)") I proposed in my GSoc application, there is an example where a rotation is performed on a qubit to bring it to a desired final state.  
This task is commonly referred to as **state transfer**.  
The paper also includes the code on their [GitHub](https://github.com/frankschae/A-differentiable-programming-method-for-quantum-control/blob/master/qubit/reinforce/env_qubit.py).  
They created the qubit environment from scratch and defined their algorithm with their policy (consisting of three neural networks that return parameters for a Gaussian distribution, from which an action is then taken).  

In our case, the idea would be to insert the qubit into a Gymnasium environment (yes, Gymnasium allows defining custom environments) so that we can then use policies from Stable Baselines.
This will allow us to be modular and scalable: we can add other environments "wrapped" in Gymnasium and use different policies from Stable Baselines.

## Qubit environment from the paper
Let's briefly look at the qubit state transfer example by examining the authors' code.  
We are mainly interested in the file where the qubit (environment) is defined.  
It starts from a random initial state and the goal is to reach a target state, which in this case is (0,1) or $| 1 \rangle$ or $|↓\rangle$.  
To achieve this, GRAPE can be used, but we will try with RL.  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/9680bc12-ae42-4c11-ad43-ef48ceda8f3c" alt="image" width="250"/>  

For a qubit, we have the following **Drift Hamiltonian** representing the spontaneous evolution of the qubit around the z-axis:
$H_0 = \frac{w}{2}\sigma_z$  

Then there is a **Control Hamiltonian** $H_1 = \sigma_x$ representing the external control action applied to the qubit. It causes a rotation around the x-axis, the action is weighted by the control parameter $\alpha$.  
$\alpha$ is the control action (e.g., a magnetic field) and is limited to a specific range $[-u_{max}, +u_{max}]$.  

The **total Hamiltonian** of the system is the sum of the two:  
$H = H_0 + \alpha H_1$  

In the code, they are multiplied by a small time step $dt$ to then numerically solve the Schrödinger equation, for example, using the **Heun** method.  
The evolution of the qubit state is described by the Schrödinger equation:  
$i\hbar \frac{\partial}{\partial t} | \Psi \rangle = H | \Psi \rangle$  
Practically, there is a loop for the various time steps and with Heun(), the new x and y components of the qubit state are calculated at each step.  

The **Fidelity** $F = |\langle \Psi | \Psi_{target} \rangle|^2 = |\langle \Psi | 1 \rangle|^2$ measures how close the current qubit state is to the target state $| 1 \rangle$.  
In the code: F = $|\langle \Psi_{target,x} | x \rangle|^2 + |\langle \Psi_{target,x} | y \rangle|^2$  
Fidelity in the code is used along with 𝛼 (the control action) to calculate the reward obtained at each time step.  

After an episode (consisting of various time steps) and collecting the rewards, the future returns (rewards-to-go) are calculated, and policy.train_net() is called to improve the policy.  

## Define qubit environment in Gymnasium
As mentioned in the previous post, in the [official documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) of Gymnasium, it is explained how to create a custom environment.  

You need to define a class that extends gym.Env

The main methods and variables to rewrite are:  

**\_init_** where we will define the main parameters  
observation_space: defines the space of observable states, i.e., the shape and range of values that the environment can return as observation  
action_space: To define which actions (𝛼) the agent can take (𝛼 -> from -u_max, +u_max)    

**observation_space**  
The observations returned by 'reset' and 'step' are valid elements of 'observation_space'.  
From the [documentation](https://gymnasium.farama.org/api/spaces/#fundamental-spaces), it is recommended to use one of their Fundamental Spaces (for example 'Box' for continuous spaces, vectors of real numbers).  

**action_space**  
The Space object corresponding to valid actions, all valid actions should be contained within the space.  
For this as well, it is advisable to use a Fundamental Space like 'Box'.  

**step**(self, action): It accepts an action, computes the state of the environment after applying that action, and returns the 5-tuple (observation, reward, terminated, truncated, info).  
Then we can check whether it is a terminal state and set the "terminated" signal accordingly.  

**reset**(self): Resets the environment to the initial state and returns a tuple of the initial observation.  
reset should be called whenever a "terminated" signal has been issued.  

**close**(self): Optional, to close the environment.

For more information on these methods, see the [documentation](https://gymnasium.farama.org/api/env/#gymnasium.Env).  

Furthermore, to make the code more integrated with QuTiP, we can use the following:  

**mesolve()**
Instead of Heun function.  
mesolve is a function in the QuTiP library used to solve the Lindblad master equation for open quantum systems or the von Neumann equation for closed systems.  
The master equation describes the dynamics of a quantum system interacting with an external environment, which can include effects of decoherence and dissipation.  
The function returns an instance of qutip.solver.Result.  
The attribute 'expect' in 'result' is a list of expectation values for the operators included in the list in the fifth argument of mesolve().  
If the list of operators is empty, it can return a list of state vectors (a list based on the time instances 'tlist').  
For details, refer to the [documentation](https://qutip.org/docs/4.0.2/guide/dynamics/dynamics-master.html).  

Represent ket states with QuTiP's **basis()** function.

**rand_ket()** is used to create the random initial state.  

For **fidelity** we can use the fidelity(A, B) function from QuTiP.  
It works for both pure states (kets) and density matrices.  
More information [here](https://qutip.org/docs/4.0.2/apidoc/functions.html#qutip.metrics.fidelity).  

### State transfer - Hadamard rotation
As a first example, let's see how to make the agent learn how to start from an initial state |0> and arrive at the state |+>, corresponding to applying the Hadamard gate to the initial state.  
This is code you can copy and run, the code seems very long but in reality many lines are for debugging.

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
        self.u_max = 2 * np.pi * 0.3  # (300 MHz)
        self.w = 2 * np.pi * 3.9  # (GHz)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Observation space, |v> have 2 real and 2 imaginary numbers -> 4

        # time for mesolve()
        self.time_end = 0.2

        # threshold for fidelity to consider the target state reached
        self.fidelity_threshold = 0.99 

        self.current_step_in_the_episode = 0
        self.max_steps = 400    # max episode length (max number of steps for one episode)

        # Reward parameters
        self.C1 = 0.016
        #self.C2 = 3.9e-06
        self.step_penalty = 0.001  # step penalty parameter 

        
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
        alpha = action * self.u_max # the action is limited between -u_max , +u_max. 
        H = self.H_0 + alpha * self.H_1

        result = qu.mesolve(H, self.state, [0, self.time_end])
        self.state = result.states[-1] # result.states returns a list of state vectors (kets), is a a Qobj object. let's take the last one.

        fidelity = qu.fidelity(self.state, self.target_state)
        #reward = fidelity
        reward = self.C1 * fidelity - self.step_penalty * self.current_step_in_the_episode
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

    def create_init_state(self, noise=False, random=True):
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
    model.learn(total_timesteps=200000) #200000
 
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
    plt.tight_layout()
    #print(f"The actions of episode{num_episodes - 1}\n {env.actions[num_episodes - 1]}")   #to see the numerical values ​​of the shares


    # Test the model
    num_tests = 10 # Number of tests to perform
    max_steps = 400 # max number of steps in eatch test
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
I would like to tell you about **two main problems** encountered in arriving at this version of the code:  

First, always initially leave the **truncated signal** equal to False.  
Using the truncated signal allowed the algorithm to learn even if the episodes were not complete, which on complex and long problems can be very useful.  
Using this signal, I immediately noticed a constant and regular increase in fidelity during the training, as if I were actually learning.  
However, during the tests, this never went beyond, for example, 70% or 80%, this brings us to the second point: the **reward function**.  
Initially I used a reward = const. * fidelity (with const for example +0.01) and fidelity given by the overlap between initial and final state.  
After many tries varying parameters like max_steps and total_timesteps, I plotted the number of steps used in each episode and set the maximum length to 400, the output I got was similar to:  

[... 98, 108, 135, 174, 82, 84, 259, 141, 60, 82, 400, 191, 58, 75, 111, 300, 157, 73, 34, 43, 78, 153, 49, 77, 100, 153, 153, 93, 81, 316, 80, 49, 53, 130, 258, 149, 75, 157, 320, 145, 74, 66, 36, 124, 50, 118, 54, 400, 106, 273, 8, 74, 101, 189, 41, 338, 95, 102, 58, 400, 255, 157, 93, 79, 200, 56, 400, 41, 345, 79, 196, 400, 200, 400, 400, 360, 142, 400, 99, 400, 133, 400, 400, 400, 400, 400, 400, 400, 66, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 203, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, ...]  

Which was counterintuitive... over time the algorithm should have taken fewer and fewer steps to reach the target state (as seen in my previous post).  
Furthermore, initially the fidelity was more random (as one would expect) and sometimes reached 99% but then stabilized and increased during the training until it reached around 80% but never went beyond that. This is because it did not reach the target state and this can be observed on the Bloch sphere.  

I finally realized that the problem was the **reward function!**  
Giving only positive rewards (based on fidelity) did not incentivize finding the best path. The algorithm had learned that by taking more time (400 maximum steps) and still going in the direction of the target state it obtained a greater cumulative reward.  
The solution was to add a penalty term to the reward: subtract an amount proportional to the number of steps used in the episode.  

Some might argue that even with the previous reward = const. * fidelity the algorithm should have learned to arrive at the target state, as states in which it reached a fidelity of over 99% could be observed in the training episodes.  
However, these cases that could be observed at the beginning of the training (because the algorithm also acts randomly to carry out exploration) were reached in much less than 400 maximum steps and the reward obtained overall could be lower overall compared to using all and 400 steps and still go in the right direction.  
And let's not forget that the algorithm only learns by observing the cumulative rewards of the episodes, it doesn't know what a "target state" is.  

Some might argue that even with the previous reward = const. * fidelity the algorithm should have learned to arrive at the target state, as states in which it reached a fidelity of over 99% could be observed in the training episodes.  
However, these cases that could be observed at the beginning of the training (because the algorithm also acts randomly to carry out exploration) were reached in much less than 400 maximum steps and the reward obtained overall could be lower overall compared to using all and 400 steps and still go in the right direction.  
And let's not forget that the algorithm only learns by observing the cumulative rewards of the episodes, it doesn't know what a "target state" is.  

Another objection, however, could be:  
the algorithm could then learn to reach the target state in a maximum of 400 steps so as to use the maximum number of steps (therefore having more positive rewards) and also have increasingly greater fidelity.
This is probably true given a much longer training time.
By instead giving a reward that also includes a penalty term proportional to the number of steps used, we give an incentive to reach the target state faster and therefore we can see the target state being reached in less time.

By running the code, finally various tests are performed after the model is trained, this is the Bloch sphere with initial state (up), final state (orange arrow) and target state |+>. The fidelity obtained in this specific test is > 99.8%

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/13cac917-5be8-4b70-85d3-b78bd9e8a99b" alt="image" width="300"/>  

Some useful data from this training:  
Number of Terminated episodes 6769, Truncated episodes 9.  
An example of the steps used in each episode:  
[92, 172, 112, 173, 109, 23, 64, 400, 170, 51, 171, 133, 128, 120, 28, 321, 64, 320, 400, 185, 241, 97, 104, 179, 35 , 69, 81, 94, 283, 121, 63, 187, 7, ... 5, 115, 67, 49, 44, 58, 44, 47, 123, 140, 39, 40, 63, 39, 63, 35, 97, 40, ... 54, 31, 54, 44, 36, 35, 73, 50, 40, 31, 45, 40, 35, 30, 36, 45, 49, ... 21, 26, 21, 26, 26, 31, 26, 31, 31, ..., 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]  
As you can see, the number of steps used for each episode decreases until the minimum number is found.  

This is the graph of the actions (alpha) taken during one of the last episodes of the training, in which it already reached fidelity > 99%  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/c14e58df-5a1c-43ef-a403-de3499f47928" alt="image" width="800"/>  

Note that the action chosen at the beginning of the step is used to evolve the system with mesolve(), so until the next step we can say that the value of the action remains constant.  
This is represented graphically using the step() method of matplotlib.  

This is the graphical representation of the final states reached at each step (16 in total)  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/393afd07-ef61-4041-bc3c-d755e1d93026" alt="image" width="300"/> 

During the training with the total_timesteps = 200000, it could be noted in the terminal, thanks to the information from Gymnasium, that the average length of the episode was still decreasing even towards the end of the training.  
By trying to extend the total duration to total_timesteps = 250000, in fact, what you can notice is that the maximum fidelity has remained approximately unchanged, greater than 99%, but now the algorithm has found the minimum number of steps (13) that it could use, keeping the other parameters of the qubit fixed.  

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/51adc6e4-5ea6-45e3-9408-bd944e35c5e7" alt="image" width="800"/>  

Furthermore, if you set the parameter random = True of create_init_state() in practice the initial state is chosen randomly between |1> and |0>  
By running the train you can notice that in the final tests a fidelity > 99% is always achieved and the algorithm correctly learns to reach the target state starting from one of the two initial states.  

### State transfer - Not Gate
You might think that to perform a different state transfer (for example Not Gate) you just need to compile the code above, modify the target state with |1> and start the training.  
If you try to run the code like this you will notice that it doesn't work, the algorithm learns almost nothing during training (fidelity very different in each episode, it didn't grow constantly), almost all the episodes end with a truncated signal (because they reach the maximum number of steps) and the tests all have very low and different fidelity. 

How is it possible?  

If we look at the number of steps used in each episode:  
[400, 400, 141, 400, 353, 341,...400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400]  
Furthermore, by plotting the actions taken towards one of the last episodes, they appear almost random.

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/554378ac-5bf7-4972-a309-887fffff7af4" alt="image" width="800"/>  

We can see similar behavior to that seen before, and it can be corrected by adjusting the reward function.  
In fact, in this task the target state is further away from the initial one, consequently I will have more steps that will give me negative cumulative rewards, so much so that the positive part of the reward given by fidelity becomes irrelevant.  
By trying to give more importance to the positive part, for example by a factor of 10 (C1=0.16) than before, the training now seems much better and the algorithm seems to be learning.  
(keeping the other parameters the same)

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/3a51f216-694b-4a70-9e80-7e771fd3da46" alt="image" width="300"/>

Fidelity during the train: initially fluctuates a lot, but as it progresses it stabilizes and increases until it almost always reaches > 99%
8 out of 10 tests achieved a fidelity > 99%
The number of steps used for the episodes fluctuates a bit towards the end:
[ 400, 347, 400, 400, 149, 400, 400, 400, 400, 400, 193, ... 290, 269, 56, 144, 400, 80, 135, 225, 290, 269, 97, 204, 225, 205, 247, 355, 169, 158, 90]
This means that it probably still needs some training, which makes sense since we have to reach a further state than previously with Hadamar rotation.  

Action plot towards one of the last episodes:

<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/9d379941-a22c-4172-9d81-8047f09d0852" alt="image" width="800"/>  

Here is the code almost identical to before:  

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
        self.u_max = 2 * np.pi * 0.3  # (300 MHz)
        self.w = 2 * np.pi * 3.9  # (GHz)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Observation space, |v> have 2 real and 2 imaginary numbers -> 4

        # time for mesolve()
        self.time_end = 0.2

        # threshold for fidelity to consider the target state reached
        self.fidelity_threshold = 0.99 

        self.current_step_in_the_episode = 0
        self.max_steps = 400    # max episode length (max number of steps for one episode)

        # Reward parameters
        self.C1 = 0.16
        #self.C2 = 3.9e-06
        self.step_penalty = 0.001  # step penalty parameter 

        
        # Target state |1>
        self.target_state = qu.basis(self.dim, 1)
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
        alpha = action * self.u_max # the action is limited between -u_max , +u_max. 
        H = self.H_0 + alpha * self.H_1

        result = qu.mesolve(H, self.state, [0, self.time_end])
        self.state = result.states[-1] # result.states returns a list of state vectors (kets), is a a Qobj object. let's take the last one.

        fidelity = qu.fidelity(self.state, self.target_state)
        #reward = fidelity
        reward = self.C1 * fidelity - self.step_penalty * self.current_step_in_the_episode
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

    def create_init_state(self, noise):
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
    model.learn(total_timesteps=200000) #200000
 
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
    plt.tight_layout()
    #print(f"The actions of episode{num_episodes - 1}\n {env.actions[num_episodes - 1]}")   #to see the numerical values ​​of the shares

    # Test the model
    num_tests = 10 # Number of tests to perform
    max_steps = 400 # max number of steps in eatch test
    figures = []  # List to store the figures
    target_state = qu.basis(env.dim, 1)

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
