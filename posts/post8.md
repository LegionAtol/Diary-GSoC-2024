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
It starts from a random initial state and the goal is to reach a target state, which in this case is (0,1) or $| 1 \rangle$ or $|↑\rangle$.  
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
observation_space: defines the space of observable states, i.e., the shape and range of values that the environment can return as observation (x, y in the code, the state of the qubit).  
x, y have the shape: (n_par, dim, 1) dim is the dimension of the Hilbert space (=2).  
action_space: To define which actions (𝛼) the agent can take (𝛼 -> from -u_max, +u_max)  
𝛼 should have the form (n_par, 1, 1).  
n_par is the number of parallel simulations, to keep it simple, we keep it 1.  

**step**(self, action): It accepts an action, computes the state of the environment after applying that action, and returns the 5-tuple (observation, reward, terminated, truncated, info).  
Then we can check whether it is a terminal state and set the "terminated" signal accordingly.  

**reset**(self): Resets the environment to the initial state and returns a tuple of the initial observation.  
reset should be called whenever a "terminated" signal has been issued.  

**close**(self): Optional, to close the environment.

For more information on these methods, see the [documentation](https://gymnasium.farama.org/api/env/#gymnasium.Env.step).  

> [!CAUTION]
> The code is still under construction

```python
import gymnasium as gym
from gymnasium import spaces
import qutip as qu
import numpy as np

# PyTorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GymQubitEnv(gym.Env):
    def __init__(self, seed):
        super(GymQubitEnv, self).__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(1,1,1), dtype=np.float32) # Box to define continuous actions space. 𝛼(action) is from -u_max to +u_max
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2 * dim,), dtype=np.float32)
        
        # Initialize qubit parameters
        self.dt = 0.001  # time step
        self.n_substeps = 20 # substeps in ODESolver
        self.dim = 2 # dimension of Hilbert space
        self.u_max = 2*np.pi*0.3 #(300 MHz)
        self.w = 2*np.pi*3.9  # (GHz)

        # for the reward
        self.C1 = 0.016
        self.C2 = 3.9e-06
        
        # Target states and Hamiltonians
        target_state = np.array([0.0, 1.0]) # |1>
        self.target_x = torch.as_tensor(np.real(target_state), dtype=torch.float, device=device).view(1,1,self.dim)
        self.target_y = torch.as_tensor(np.imag(target_state), dtype=torch.float, device=device).view(1,1,self.dim)
        H_0 = self.w/2*qu.sigmaz()
        H_1 = qu.sigmax()
        self.H_0_dt = torch.as_tensor(np.real(H_0.full())*self.dt, dtype=torch.float, device=device)
        self.H_1_dt = torch.as_tensor(np.real(H_1.full())*self.dt, dtype=torch.float, device=device)

        self.state = None
        self.seed(seed)

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def step(self, action):
        alpha = torch.clamp(torch.tensor(action, dtype=torch.float, device=device), min=-self.u_max, max=self.u_max).view(1, 1, 1)
        x, y = self.state
        for _ in range(self.n_substeps):
            H = self.H_0_dt + alpha * self.H_1_dt
            x, y = self.Heun(x, y, H)
        fidelity = (torch.matmul(self.target_x, x)**2 + torch.matmul(self.target_x, y)**2).squeeze().item()
        abs_alpha = (alpha**2).squeeze().item()
        reward = self.C1*fidelity - self.C2*abs_alpha
        #if t == max_episode_steps+1:
        #    reward += C3*fidelity
        self.state = x, y
        terminated = False
        truncated = False
        observation = self._get_obs()
        return observation, reward, terminated, truncated, {}

    def reset(self, noise=True):
        psi_x, psi_y = create_init_state(noise, 1)
        self.state = psi_x.view(1, self.dim, 1), psi_y.view(1, self.dim, 1)
        return self._get_obs()

    def _get_obs(self):
        x, y = self.state
        return torch.cat((x, y), 1).transpose(1, 2).cpu().numpy().flatten()

    def Heun(self, x, y, H_dt):
        f_x, f_y = torch.matmul(H_dt, y), -torch.matmul(H_dt, x)
        x_tilde, y_tilde = x + f_x, y + f_y
        x, y = x + 0.5 * (torch.matmul(H_dt, y_tilde) + f_x), y + 0.5 * (-torch.matmul(H_dt, x_tilde) + f_y)
        return x, y

    def create_init_state(noise, n_par):
        dim = 2
        psi_x, psi_y = torch.zeros((n_par,dim), dtype=torch.float, device=device), torch.zeros((n_par,dim), dtype=torch.float, device=device)
        if noise:
            # Note that theta [0, 2pi] is biased towards the poles
            # theta	=	cos^(-1)(2v-1) with v on [0,1]
            theta = torch.acos(torch.zeros((n_par,), dtype=torch.float, device=device).uniform_(-1.0, 1.0))
            phi = torch.zeros((n_par,), dtype=torch.float, device=device).uniform_(0.0, 2*np.pi)
    
            psi_x[:, 0] += torch.cos(theta / 2) # real part of coefficient of |up>
            psi_x[:, 1] += torch.sin(theta / 2)*torch.cos(phi) # real part of coefficient of |down>
    
            psi_y[:, 0] += torch.zeros_like(theta)  # imag part of coefficient of |up>
            psi_y[:, 1] += torch.sin(theta / 2)*torch.sin(phi)
        else:
            psi_x[:, 0], psi_y[:, 0] = 1, 0 #1, 0 # |up>
    
        return psi_x, psi_y

if __name__ == '__main__':
    env = GymQubitEnv(seed=100)
    n_par = 3 # number of parallel simulations
    state = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(reward)
    exit()
```
In another file, we can call the environment and use a Stable Baselines3 policy to train and test the model:  

```python
import env_qubit    # env_qubit.py environment file
from stable_baselines3 import PPO

env = GymQubitEnv(seed=42)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs = env.reset()
```
