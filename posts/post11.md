# 2 Qubits Example - CNOT
So far, we have used a single qubit to try executing the Hadamard gate; now, let's try using two qubits to execute the CNOT gate.  
In practice, there is a control qubit and a target qubit.  
If the control qubit is in the state |0⟩, the target qubit remains unchanged.  
If the control qubit is in the state |1⟩, the CNOT gate flips the state of the target qubit.  
The matrix that describes the CNOT gate is 4x4 because it acts on a two-qubit state, unlike the Hadamard rotation matrix, which was 2x2.
To summarize:  
CNOT⋅|00⟩=|00⟩  
CNOT⋅|01⟩=|01⟩  
CNOT⋅|10⟩=|11⟩  
CNOT⋅|11⟩=|10⟩  

Looking at the functions **cy_grape_unitary()**, **cy_grape_inner()** and the examples for CNOT in the QuTiP [documentation](https://nbviewer.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-cnot.ipynb), we can immediately notice a greater complexity than in the single qubit case.  
$H_{ops}$ is a list of control Hamiltonians, no longer just, for example, $\sigma_x$ or $\sigma_y$, but now tensor products to describe the interactions on both qubits.  
The drift Hamiltonian is H0.  

The goal of the optimization is to minimize the difference between the obtained unitary operator $U_f$  and the target U=CNOT.  
In other words, we want the overlap between P and Q to be as close to one as possible, indicating that  is correctly evolving towards U.  
$H_t$ is the sum of H0 and $H_{ops}[j]$ multiplied by the control fields u.  
$U_{list}$ is a list of matrices (or NumPy arrays) representing the time evolution operators for each time interval dt.  

$U_f(t)$ represents the evolution operator that describes the system's evolution from the initial time t=0 to a time t.  
It is a cumulative unitary operator.  
$U_{f,list}$ contains the matrices $U_f(t)$ for each time point t.  

$U_b(t)$ is the cumulative unitary operator (called the backward propagator) that describes the system's evolution from the final time T to a time t.  
$U_{b,list}$ contains the matrices $Ub(t)$ for each time point.  

P is the unitary matrix that combines the backward propagator $U_b$ and the target operator U. 
P represents how the target operator U compares with the backward evolution of the system up to that point.

Q is the matrix that includes the effect of the control operator $H_{ops}[j]$ on the forward evolution $U_f$.  
Q represents the influence of a small change in control (associated with the operator $H_{ops}[j]$) on the system's evolution.  

Process:
+ Calculate $U_f$  and $U_b$  for each time point.  
+ Combine the backward evolution with the target operator U to obtain P.
+ Combine the forward evolution with the control operator $H_{ops}[j]$ to obtain Q.
+ Use the overlap between P and Q to calculate the gradient (du) of the error.
+ Update the control fields $u(t)$ using the gradient.
Repeat the process. M is the number of time steps for each iteration, used to divide the total evolution time of the system into M steps. There are R iterations in total.

In the function cy_grape_inner(), the controls u are updated.  
To calculate how to update the control fields, they use $du=-overlap(P,Q)$  
$overlap = |\frac{tra(U_{targ}^* U_{final})}{N}|$  
This overlap could be our reward function from an **RL perspective**.  
The closer it is to 1, the more similar the two operators are.  
Note that in cy_grape_unitary() there is first a loop on m to calculate all the propagators $U_f$ and $U_b$ (and their respective lists) in order to then calculate the gradients in cy_grape_inner() which again loops on m.  
From an RL perspective there is no need to have gradients, the RL agent will decide how to update future actions, so $U_{f,list}$ and $U_{b,list}$ will not be needed.  

The following code is an example of two qubits where the RL agent tries to learn the CNOT gate.  
It is simplified because it does not yet exploit all the features of Qutip and the various modules.  

> [!WARNING]  
> The code is still under development  

```python
import gymnasium as gym
from gymnasium import spaces
import qutip as qt
from qutip_qoc import *
from qutip_qtrl.cy_grape import *
from qutip.qip.operations import *
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time

class GymMultiQubitsEnv(gym.Env):
    def __init__(self, H0, H_ops, U_target, tlist, R, fid_err_targ):
        super(GymMultiQubitsEnv, self).__init__()
        self.H0 = H0
        self.H_ops = H_ops
        self.U_target = U_target
        self.T = tlist[-1]
        self.dt = self.T / len(tlist)
        self.max_steps = len(tlist)
        self.max_iter = R
        self.N = U_target.shape[0]
        self.current_step = 0
        self.U_current = qt.tensor(qt.qeye(2), qt.qeye(2))      # current operator
        self.controls = np.zeros(len(self.H_ops))
        self.fid_err_targ = fid_err_targ

        self.state_dim = 2 * self.N**2  # (2*N^2 dim,  real and immaginary part)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.state_dim,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.H_ops),))
    
    def reset(self, seed=None):
        self.current_step = 0
        #self.U_current = qt.qeye(self.N)
        self.U_current = qt.tensor(qt.qeye(2), qt.qeye(2))
        self.controls = np.zeros(len(self.H_ops))
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Get the density matrix
        #density_matrix = self.U_current @ self.U_current.dag()
        real_part = np.real(self.U_current.full())
        imag_part = np.imag(self.U_current.full())
        obs = np.concatenate((real_part.flatten(), imag_part.flatten()))
        return obs.astype(np.float32)
    
    # evolution of the operator U for a time interval dt under the influence of H_t
    def evolve_system(self, H0, H_ops, controls, dt, current_U):
        H_t = H0 + sum([controls[j] * H_ops[j] for j in range(len(H_ops))])
        U_next = (-1j * H_t * dt).expm() @ current_U
        return U_next

    def step(self, action):
        # Update controls based on the action
        alpha = action * 0.1    # -0.1 , 0.1
        self.controls += alpha
        
        # Evolve the system for the current time step
        self.U_current = self.evolve_system(self.H0, self.H_ops, self.controls, self.dt, self.U_current)
        
        self.current_step += 1

        overlap = cy_overlap(self.U_current.data, self.U_target.data)   # overlap of the two operators
        abs_overlap = np.abs(overlap)
        reward = abs_overlap - self.current_step*0.0001
        terminated = abs_overlap >= 1 - self.fid_err_targ
        truncated = self.current_step >= self.max_steps

        if terminated:
            print(f"total steps for this episode{self.step}")

        return self._get_obs(), reward, bool(terminated), bool(truncated), {}
    
    def train(self):
        # Check if the environment follows Gym API
        check_env(self, warn=True)

        # Create the model
        model = PPO('MlpPolicy', self, verbose=1)

        # Train the model
        model.learn(total_timesteps = self.max_iter * self.max_steps)



if __name__ == '__main__':
    
    # Define the problem

    # single qubit control operators
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    identity = qt.qeye(2)

    # two qubit control operators
    i_sx, sx_i = qt.tensor(sx, identity), qt.tensor(identity, sx)
    i_sy, sy_i = qt.tensor(sy, identity), qt.tensor(identity, sy)
    i_sz, sz_i = qt.tensor(sz, identity), qt.tensor(identity, sz)

    w = 2 * np.pi * 3.9  # (GHz)
    H0 = 0 * np.pi * (qt.tensor(qt.sigmax(), qt.identity(2)) + qt.tensor(qt.identity(2), qt.sigmax()))
    #H0 = w * (i_sz + sz_i) + delta * i_sz * sz_i

    H_ops = [i_sx, i_sy, i_sz, sx_i, sy_i, sz_i]
    #Hc = [qt.liouvillian(H) for H in Hc]

    U_target = cnot()
    #U_initial = qt.qeye(U_target.shape[0])
    
    u_limits = None
    
    T = 2 * np.pi 
    tlist = np.linspace(0, T, 500)

    R = 500     # max number of eopchs/episodes for training
    fid_err_targ = 0.01

    # create the environment
    env = GymMultiQubitsEnv(H0, H_ops, U_target, tlist, R, fid_err_targ)

    env.train()
```
