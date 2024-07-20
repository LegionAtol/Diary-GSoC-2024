# Qubit Example
## Integration with qoc module: Objective and Result
I created the following very simple code, it is a trivial version of the qubit where only 10 steps are performed.  
This code is used to experiment and understand how to use the Objective and Result classes and other useful methods in the qoc module. 

> [!IMPORTANT]  
> The code is under development.

```python
import numpy as np
import qutip as qt
from qutip_qoc.objective import Objective
from qutip_qoc.result import Result
import time

class Qubit():
    def __init__(self, objectives):
    
        self.Hd_lst, self.Hc_lst = [], []
        if not isinstance(objectives, list):
            objectives = [objectives]
        for objective in objectives:
            # extract drift and control Hamiltonians from the objective
            self.Hd_lst.append(objective.H[0])
            self.Hc_lst.extend([H[0] if isinstance(H, list) else H for H in objective.H[1:]])

        print("H drift: \n", self.Hd_lst[0])
        print("H control \n", self.Hc_lst[0])

        # extract initial and target state from the objective
        self.initial_state = objectives[0].initial
        self.target_state = objectives[0].target

        #initila_guess_controls = [np.array([0.3])]  # Initial guess for control parameter

        self.result = Result(
            objectives=objectives,
            time_interval=None, #vedi qutip_qoc._TimeInterval
            start_local_time=time.localtime(),  # initial optimization time
            end_local_time = None,  # final optimization time
            #_total_seconds = None,   # total time taken to complete the optimization
            n_iters = 0,           # Number of iterations(episodes) until convergence 
            iter_seconds = [],    # list containing the time taken for each iteration(episode) of the optimization
            #_guess_controls = initila_guess_controls,
            #guess_params=None,     ??? there is no documentation on Github
            #_final_states = [],      # List of final states after the optimization. One for each objective.
            #_optimized_H
            #optimized_params = [],  #list of ndarray. List of optimized parameters
            var_time=False,         # Whether the optimization was performed with variable time
        )

    def step(self):
        alpha = 0.11 #control parameter

        H = self.Hd_lst[0] + alpha * self.Hc_lst[0]
        step_result = qt.mesolve(H, self.initial_state, [0, 0.01])
        state =  step_result.states[-1]

        infidelity = 1 - qt.fidelity(state, self.target_state)
        time_diff = time.mktime(time.localtime()) - time.mktime(self.result.start_local_time)
        self.result.iter_seconds.append(time_diff)
        self.result.n_iters += 1  

        if self.result.n_iters == 10 or infidelity < 0.01:
            self.result.message = "Optimization finished"
            self.result.end_local_time = time.localtime()
            # total_seconds is handled automatically with @property
            self.result.n_iters = len(self.result.iter_seconds)  # or number of steps used
            self.result.optimized_params = np.array([alpha])    
            #self.result.final_states.append(state)  # handled automatically with @property
            return True
        
        return False
    


if __name__ == "__main__":
    # Define the problem (input)
    w = 2 * np.pi * 3.9  # (GHz)

    H_0 = w / 2 * qt.sigmaz()
    H_1 = qt.sigmax()

    initial_state = qt.basis(2,0)
    target_state = (qt.gates.hadamard_transform() * qt.basis(2, 0)).unit()  # Hadamard applied to |0>

    def pulse(t, p):
        #return p * np.sin(omega * t)
        return p

    objective = Objective(
            initial = initial_state,    #initial state
            #H = [H_0, [H_1]],          # the Hamiltonians
            H=[H_0, [H_1, lambda t, p: pulse(t, p)]],    # the Hamiltonians
            target = target_state    #target state
            )

    qubit = Qubit(objective)

    for step in range(10):
        if qubit.step():
            break       # if it ends before 10 steps
    
    print(qubit.result)
    print(qt.fidelity(qubit.target_state, qubit.result.final_states[0]))
```

