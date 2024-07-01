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
In the [official documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) of Gymnasium, it is explained how to create a custom environment.  

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
  
   
> [!CAUTION]
> The code is still under construction

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
        #self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(1,), dtype=np.float32)  # Continuous action space from -u_max to +u_max
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(1,), dtype=np.float32)  # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Observation space, |v> have 2 real and 2 imaginary numbers -> 4

        # time for mesolve()
        self.times = np.linspace(0, 1, 100)

        # threshold for fidelity to consider the target state reached
        self.fidelity_threshold = 0.99 

        self.current_step_in_the_episode = 0
        self.max_steps = 150

        # Reward parameters
        self.C1 = 0.016
        self.C2 = 3.9e-06
        
        # Target state |1>
        self.target_state = qu.basis(self.dim, 1)
        
        # Hamiltonians
        self.H_0 = self.w / 2 * qu.sigmaz()
        self.H_1 = qu.sigmax()

        #for debugging
        self.episode_reward = 0 
        self.rewards = []
        self.fidelities = []

        self.state = None
        self.seed = None

    #def seed(self, seed=None):
    #    np.random.seed(seed)

    def step(self, action):
        alpha = action * self.u_max # the action is limited between -u_max , +u_max.
        H = self.H_0 + alpha * self.H_1

        result = qu.mesolve(H, self.state, self.times)
        self.state = result.states[-1] # result.states returns a list of state vectors (kets), is a a Qobj object. let's take the last one.

        fidelity = qu.fidelity(self.state, self.target_state)
        reward = self.C1 * fidelity - self.C2 * (alpha ** 2)
        self.current_step_in_the_episode += 1
        terminated = self.current_step_in_the_episode >= self.max_steps or fidelity >= self.fidelity_threshold

        #for debugging
        print(f"Step {self.current_step_in_the_episode}, Fidelity: {fidelity}")
        self.episode_reward += reward
        if terminated:
            self.fidelities.append(fidelity) # keep the final episode fidelity

        observation = self._get_obs()
        
        truncated = False

        reward = float(reward.item())  # Ensure the reward is a float
        #print("\n the reward is:",reward,"\n")
        
        return observation, reward, bool(terminated), bool(truncated), {"state": self.state}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.state = self.create_init_state(noise=True)
        
        self.rewards.append(self.episode_reward) # keep the episode rewards
        self.episode_reward = 0  # Reset the episode reward
        self.current_step_in_the_episode = 0  # Reset the step counter
        return self._get_obs(), {}

    # if state=(p q)' with p = a + i*b and q = c + i*d -> return [a, b, c, d]
    def _get_obs(self):
        rho = self.state.full().flatten() # to have state vector as NumPy array and flatten into one dimensional array.[a+i*b c+i*d]
        obs = np.concatenate((np.real(rho), np.imag(rho)))
        return obs.astype(np.float32) # Gymnasium expects the observation to be of type float32

    def create_init_state(self, noise):
        if noise:
            # initial random ket state
            init_state = qu.rand_ket(self.dim)
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
    model.learn(total_timesteps=80000)

    # For debugging
    for i, (r, f) in enumerate(zip(env.rewards, env.fidelities), start=1):
        print(f"Rewards for episode {i}: {r}")
        if i % 50 == 0:
            avg_fidelity = np.mean(env.fidelities[i-50:i])
            print(f"Episode {i}, Avg fidelity of last 50 episodes: {avg_fidelity}")

    # Test the model
    num_tests = 10 # Number of tests to perform
    max_steps = 150 # max number of steps in eatch test
    figures = []  # List to store the figures
    target_state = qu.basis(2, 1)

    for test in range(num_tests):
        print(f"\nTest {test + 1}")
        obs, _ = env.reset()  # Reset the environment to get a random initial state
        initial_state = env.state  # Save the initial state
        
        for _ in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)  # Get action from the model
            obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
            final_state = info["state"]  # Get the final state from the environment
            if terminated or truncated:  # Check if the episode has ended
                # Compute fidelity between final state and target state
                fidelity = qu.fidelity(final_state, target_state)
                print("Final Fidelity:", fidelity)
                # Visualize on the Bloch sphere
                b = qu.Bloch()
                b.add_states(initial_state)
                b.add_states(final_state)
                fig = plt.figure()  # Create a new figure
                b.fig = fig  # Assign the figure to the Bloch sphere
                b.render()  # Render the Bloch sphere
                figures.append(fig)  # Store the figure in the list
                break  # Exit the loop if the episode has ended         
    # Show all figures together
    plt.show()
```
