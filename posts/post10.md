# Qubit Example
## Integration with qoc module: Objective and Result  

To adapt my code to the qoc package from QuTiP, I decided to create a simpler version of the qubit and use methods and portions of code from qoc.  
Initially, I had to thoroughly review the code in pulse_optim.py, objective.py, result.py, and _optimizer.py (you can find all of them in the qutip-qoc GitHub repository).  

As **input** to my code I could have:
+ objective: An instance of the Objective class.
+ control_parameters: A dictionary to define the limit values of the parameters to be optimized.
+ tlist: To define the maximum time of an episode and the maximum number of steps it can contain.
+ algorithm_kwargs: A dictionary for training parameters like infidelity target and the maximum number of iterations (episodes) for training.

As **output**:  
The goal is to have an instance of the Result class that contains important information about the optimization process, such as the final state, final infidelity, optimization duration, etc.  

As I mentioned, I created a simpler code to better experiment with the various classes, methods and variables of qoc.  
Specifically, the code does not involve training or learning. It executes only one episode consisting of a maximum of 10 steps to go from the initial state to the target state.  
The value of the optimal action is manually set and passed to step().  
This program is inspired by the one in the previous post: Manual Hadamard rotation with Qutip.  

```python
import numpy as np
import qutip as qt
from qutip_qoc import *
import matplotlib.pyplot as plt
import time

class Qubit():
    def __init__(self, objectives, control_parameters, tlist, algorithm_kwargs):
        
        # create time interval
        time_interval = _TimeInterval(tslots=tlist)
        self.max_episode_time = time_interval.evo_time                  # maximum time for an episode
        self.max_steps = time_interval.n_tslots                         # maximum number of steps in an episode
        self.step_duration = time_interval.tslots[-1] / time_interval.n_tslots  # step duration for mesvole()
        self.max_episodes = algorithm_kwargs["max_iter"]                # maximum number of episodes for training

        self.H = objectives.H
        self.Hd_lst, self.Hc_lst = [], []
        if not isinstance(objectives, list):
            objectives = [objectives]
        for objective in objectives:
            # extract drift and control Hamiltonians from the objective
            self.Hd_lst.append(objective.H[0])
            self.Hc_lst.append([H[0] if isinstance(H, list) else H for H in objective.H[1:]])

        self.pulse = objective.H[1][1]      # extract control function

        # extract bounds for the control pulses
        bounds = []
        for key in control_parameters.keys():
            bounds.append(control_parameters[key].get("bounds"))
        self.lbound = [b[0][0] for b in bounds]
        self.ubound = [b[0][1] for b in bounds]

        # extract initial and target state from the objective
        self.init_state = objectives[0].initial
        self.targ_state = objectives[0].target
        self.state = self.init_state                # actual state durign optimization

        self.fid_targ = 100 - algorithm_kwargs["fid_err_targ"]

        #initila_guess_controls = [np.array([0.3])]  # Initial guess for control parameter

        self.result = Result(
            objectives = objectives,
            time_interval = time_interval,
            start_local_time = time.localtime(),    # initial optimization time
            #end_local_time = None,                 # final optimization time
            #total_seconds = None,                  # total time taken to complete the optimization
            n_iters = 0,                            # Number of iterations(episodes) until convergence 
            iter_seconds = [],                      # list containing the time taken for each iteration(episode) of the optimization
            #guess_controls = None                 X
            # guess_params = None                   X
            #final_states = [],      # List of final states after the optimization. One for each objective.
            #optimized_H
            #optimized_params = [],  #list of ndarray. List of optimized parameters
            var_time=False,         # Whether the optimization was performed with variable time
        )

        #for debug
        self.kets = []

    def step(self, action):

        args = {"alpha" : action}
        H = [self.Hd_lst[0], [self.Hc_lst[0][0], lambda t, args: self.pulse(t, args["alpha"])]]
        step_result = qt.mesolve(H, self.state, [0, self.step_duration], args = args)
        self.state =  step_result.states[-1]

        infidelity = 1 - qt.fidelity(self.state, self.targ_state)
        self.result.infidelity = infidelity
        print(f"step n° {self.result.n_iters}, fidelity: {qt.fidelity(self.state, self.targ_state)}")   # for debug
        time_diff = time.mktime(time.localtime()) - time.mktime(self.result.start_local_time)
        self.result.iter_seconds.append(time_diff)
        self.result.n_iters += 1  
     
        self.kets.append(step_result.states[-1])    # for debug
        
        if self.result.n_iters >= 10 or infidelity < 0.01:
            self.result.message = "Optimization finished"
            self.result.end_local_time = time.localtime()
            # total_seconds is handled automatically with @property
            self.result.n_iters = len(self.result.iter_seconds)         # or number of steps used
            self.result.optimized_params = np.array([args["alpha"]])    
            #self.result._final_states.append(state)             # final_states handled automatically with @property
            self.result._final_states = (self.result._final_states if self.result._final_states is not None else []) + [self.state]  # TODO: see qoc Result()
            return True
        
        return False
    
    def training(self):
        for episode in range(self.max_episodes):    # model.train() gym
            for step in range(self.max_steps):      
                if qubit.step(action=11.3):    
                    break               # if it ends before 10 steps
       
       # Displays the states of the steps
        bloch = qt.Bloch()
        for ket in self.kets:
            bloch.add_states(ket)
            bloch.show()
        
        print(qubit.result)
        print("final fidelity : ",qt.fidelity(qubit.targ_state, qubit.result.final_states[0]) )
        
        plt.show()


if __name__ == "__main__":
    # Define the problem (input)
    w = 2 * np.pi * 3.9  # (GHz)

    H_0 = w / 2 * qt.sigmaz()
    H_1 = qt.sigmax()

    initial_state = qt.basis(2,0)
    target_state = (qt.gates.hadamard_transform() * qt.basis(2, 0)).unit()  # Hadamard applied to |0>

    # control function
    def pulse(t, p):
        #return p * np.sin(omega * t)
        return p    # constant fx for every t

    objective = Objective(
            initial = initial_state,    
            H=[H_0, [H_1, lambda t, p: pulse(t, p)]],
            target = target_state    
            )
    
    control_parameters = {
        "p": {
                #"guess": [1.0, 1.0, 1.0],
                "bounds": [(-13, 13)],
            }
    }

    tlist = np.linspace(0, 0.1, 10)

    algorithm_kwargs={
        "fid_err_targ": 0.01,
        "alg": "RL",
        "max_iter": 1          # max number of eopchs/episodes for training
    }

    qubit = Qubit(objective, control_parameters, tlist, algorithm_kwargs)
    qubit.training()
```
