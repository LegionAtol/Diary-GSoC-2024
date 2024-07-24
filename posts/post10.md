# Qubit Example
## Integration with qoc module: Objective and Result  
### Simple example
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

### Complete example  
The complete code of the qubit with qoc is as follows  

```python
import gymnasium as gym
from gymnasium import spaces
import qutip as qt
from qutip_qoc import *
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time

class GymQubitEnv(gym.Env):
    def __init__(self, objectives, control_parameters, tlist, algorithm_kwargs):
        super(GymQubitEnv, self).__init__()

        # create time interval
        time_interval = _TimeInterval(tslots=tlist)
        self.max_episode_time = time_interval.evo_time                  # maximum time for an episode
        self.max_steps = time_interval.n_tslots                         # maximum number of steps in an episode
        self.step_duration = time_interval.tslots[-1] / time_interval.n_tslots  # step duration for mesvole()
        self.max_episodes = algorithm_kwargs["max_iter"]                # maximum number of episodes for training
        self.total_timesteps = self.max_episodes * self.max_steps       # for learn() of gym

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
        self.fid_targ = 1 - algorithm_kwargs["fid_err_targ"]

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
            var_time = False,         # Whether the optimization was performed with variable time
        )
        
        self.dim = 2  # dimension of Hilbert space

        # Define action and observation spaces (Gym)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space from -1 to +1, as suggested from gym
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Observation space, |v> have 2 real and 2 imaginary numbers -> 4

        # To check if it exceeds the maximum number of steps in an episode
        self.current_step_in_the_episode = 0

        # Reward parameters
        self.C1 = 1    
        self.C2 = 1  # step penalty parameter
        
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
        
    def step(self, action):
        action = action[0]  # action is an array -> extract it's value with [0]
        #alpha = action * self.ubound # the action is limited between lbound , ubound. 
        alpha = ((action + 1) / 2 * (self.ubound[0] - self.lbound[0])) + self.lbound[0]   # for example, scale action from [-1, 1] to [9, 13]
        args = {"alpha" : alpha}

        H = [self.Hd_lst[0], [self.Hc_lst[0][0], lambda t, args: self.pulse(t, args["alpha"])]]
        step_result = qt.mesolve(H, self.state, [0, self.step_duration], args = args)
        self.state =  step_result.states[-1]

        fidelity = qt.fidelity(self.state, self.targ_state)
        infidelity = 1 - fidelity
        self.result.infidelity = infidelity

        action_difference_penalty = 0
        if self.episode_actions:
            action_difference_penalty = np.abs(action - self.episode_actions[-1])
        reward = self.C1 * fidelity - self.C2 - action_difference_penalty

        self.current_step_in_the_episode += 1

        terminated = fidelity >= self.fid_targ    # if the goal is reached
        truncated = self.current_step_in_the_episode >= self.max_steps  # if the episode ended without reaching the goal
        #truncated=False

        reward = float(reward.item())  # Ensure the reward is a float  

        # for debugging
        #print(f"Step {self.current_step_in_the_episode}, Fidelity: {fidelity}")
        self.episode_reward += reward
        self.episode_actions.append(action)
        if terminated or truncated:
            #self.result.n_iters += 1 
            time_diff = time.mktime(time.localtime()) - time.mktime(self.result.start_local_time)
            self.result.iter_seconds.append(time_diff)
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

    def reset(self, seed = None):
        self.state = self.create_init_state()
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
                init_state = qt.basis(self.dim, 1)  # |1>
            else:
                init_state = qt.basis(self.dim, 0)  # |0>
        else:
            init_state = self.init_state

        if noise:
            # Initial slight variations of |0>
            perturbation = 0.1 * (np.random.rand(self.dim) - 0.5) + 0.1j * (np.random.rand(self.dim) - 0.5) # to get something like: [0.03208387-0.01834318j 0.0498474 -0.0339512j ]
            perturbation_qobj = qt.Qobj(perturbation, dims=[[self.dim], [1]])
            init_state = init_state + perturbation_qobj
            init_state = init_state.unit()  # to ensure unitary norm
        
        return init_state
    
    def train(self):
        # Check if the environment follows Gym API
        check_env(self, warn=True)

        # Create the model
        model = PPO('MlpPolicy', self, verbose=1)

        # Train the model
        model.learn(total_timesteps = self.total_timesteps)

        self.result.message = "Optimization finished!"
        self.result.end_local_time = time.localtime()
        self.result.n_iters = len(self.result.iter_seconds)         
        self.result.optimized_params = np.array(self.actions[-1])
        self.result._final_states = (self.result._final_states if self.result._final_states is not None else []) + [self.state]  # TODO: see qoc Result()

        model.save("rl_model")

        # For debugging
        print("\n Summary of the trining:")
        for i, (r, f) in enumerate(zip(self.rewards, self.fidelities), start=1):
            #print(f"Rewards for episode {i}: {r}")
            print(f"Fidelity for episode {i}: {f}")
            if i % 50 == 0:
                avg_reward = np.mean(self.rewards[i-50:i])
                avg_fidelity = np.mean(self.fidelities[i-50:i])
                print(f"Episode {i}, Avg reward of last 50 episodes: {avg_reward}")
                print(f"Episode {i}, Avg fidelity of last 50 episodes: {avg_fidelity}\n")

        print(f"Highest fidelity achieved during training: {self.highest_fidelity}")
        print(f"Highest fidelity was achieved in episode: {self.highest_fidelity_episode}")
        print(f"Number of: Terminated episodes {self.num_of_terminated}, Truncated episodes {self.num_of_truncated}")
        print(f"Number of steps used in each episode {self.episode_steps}")
        
        # Plot actions of some episodes
        # the action chosen at each step remains constant during the evolution of the system with mesolve, 
        # therefore in the plot I represent them constant 
        num_episodes = len(self.actions)
        indices = [9, 19, num_episodes - 11, num_episodes - 1]  # 10th, 20th, (final-10)th, final episodes
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        for i, idx in enumerate(indices):
            steps = np.arange(len(self.actions[idx]))  # Create an array of step indices
            actions = self.actions[idx]  # Extract action values from the array  
            # Plot each action as a constant value over its interval
            axs[i].step(steps, actions, where='post')
            axs[i].set_title(f'Episode {idx + 1}')
            axs[i].set_xlabel('Step')
            axs[i].set_ylabel('Action')
            print(f"The actions of episode{num_episodes - 1}\n {self.actions[num_episodes - 1]}")   #to see the numerical values ​​of the shares
        plt.tight_layout()
        #print(f"The actions of episode{num_episodes - 1}\n {self.actions[num_episodes - 1]}")   #to see the numerical values ​​of the shares

    def test(self):

        # load the model
        try:
            model = PPO.load("rl_model", env=env)
        except Exception as e:
            print(f"Error during test: {e}")

        # Test the model
        num_tests = 10 # Number of tests to perform
        max_steps = self.max_steps # max number of steps in eatch test
        figures = []  # List to store the figures
        #targ_state = (qt.gates.hadamard_transform() * qt.basis(env.dim, 0)).unit()  # Hadamard applied to |0>
        targ_state = self.targ_state

        for test in range(num_tests):
            print(f"\nTest {test + 1}")
            obs, _ = self.reset()  # Reset the environment to get a random initial state
            initial_state = self.state  # Save the initial state
            #all_intermediate_states = []  # if you want to view all intermediate states
            for _ in range(max_steps):
                action, _states = model.predict(obs, deterministic=False)  # Get action from the model
                obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
                #all_intermediate_states.append(info["state"])  # Collect all final states from the steps
                if _ == max_steps-1:
                    final_state = info["state"]  # Get the final state from the environment
                    print(f"Test episode not ended! final Fidelity achived: {qt.fidelity(final_state, targ_state)}")
                if terminated or truncated:  # Check if the episode has ended
                    # Compute fidelity between final state and target state
                    final_state = info["state"]  # Get the final state from the environment
                    fidelity = qt.fidelity(final_state, targ_state)
                    print("Final Fidelity:", fidelity)
                    # Visualize on the Bloch sphere
                    b = qt.Bloch()
                    b.add_states(initial_state)
                    #b.add_states(all_intermediate_states)  # Add all states to the Bloch sphere
                    b.add_states(final_state)   # comment this out if you use b.add_states(all_intermediate_states)
                    b.add_states(env.targ_state) 
                    fig = plt.figure()  # Create a new figure
                    b.fig = fig  # Assign the figure to the Bloch sphere
                    b.render()  # Render the Bloch sphere
                    figures.append(fig)  # Store the figure in the list
                    break  # Exit the loop if the episode has ended         
        # Show all figures together
        plt.show()


if __name__ == '__main__':
    
    # Define the problem (input)
    w = 2 * np.pi * 3.9  # (GHz)
    H_0 = w / 2 * qt.sigmaz()
    H_1 = qt.sigmax()

    initial_state = qt.basis(2,0)
    target_state = (qt.gates.hadamard_transform() * qt.basis(2, 0)).unit()  # Hadamard applied to |0>

    # control function
    def pulse(t, p):
        #return p * np.sin(omega * t)
        return p    # constant fun for every t

    objective = Objective(
            initial = initial_state,    
            H = [H_0, [H_1, lambda t, p: pulse(t, p)]],
            target = target_state    
            )
    
    control_parameters = {
        "p": {
                #"guess": [1.0, 1.0, 1.0],
                "bounds": [(9, 13)],
            }
    }
    
    tlist = np.linspace(0, 0.1, 130)

    algorithm_kwargs={
        "fid_err_targ": 0.01,
        "alg": "RL",
        "max_iter": 1500          # max number of eopchs/episodes for training
    }

    # create the environment
    env = GymQubitEnv(objective, control_parameters, tlist, algorithm_kwargs)

    env.train()

    env.test()
```  

In one of the previous posts, the algorithm that found the Hadamard rotation in a single step had a fundamental requirement:  
the "optimal" duration of the step is known a priori (0.1) because I had simulated it "manually" (other code in my diary).  
The RL algorithm had max_steps=100, it starts using many step() of duration 0.1 per episode. Each step calls mesolve() of duration 0.1  
During training, the number of steps needed decreased until only one was used, because I knew that with a certain control action, it was possible to reach at least 99% fidelity in 0.1  

In the qoc package, tlist = np.linspace(…) is used to define the system evolution time.  
In RL I think there are 2 ways to interpret tlist = np.linspace(…) :  

Case 1)  
Defining for example tlist = np.linspace(0, 0.1, 100)  
It means we are setting 100 = max_steps per episode and the episode duration is 0.1  
But as mentioned, 0.1 was the duration of the step.  
So you can simply set 0.1 as the step duration and 100 as the maximum steps per episode.  
However I'm not sure if this is the way to understand tlist = np.linspace() values  
Furthermore, a problem arises:  
The user must know almost exactly the minimum/optimal duration of the step, if he puts even just 0.2 as the step time() the algorithm will find for example that 4 minimum steps will be needed and therefore a total duration greater than necessary to execute the hadamard.  
However, if you can set a step time in advance, or an optimal episode consisting of a single step, you would obtain a single value (or vector) of optimized_parameters in output (it would be an action of the RL agent).  

Case 2)  
Perhaps the 0.1 in linspace(0, 0.1, 100) can be understood as the maximum duration of the episode.  
If this were the case, then the step duration would be = 0.1 / 100 but with such a short step, the algorithm will find a better episode with a duration of 100 steps.  
However, the sum of the times of these 100 (shorter) steps should be equivalent to the initial case in which we had a single step lasting 0.1 but with the added advantage of having only one action/control value, now instead we will have 100 From a practical point of view, perhaps it may be convenient to modify the reward function in order to incentivize the algorithm to use the most similar actions possible between the various steps.  
This approach of using many smaller steps can be beneficial to the user:  
Even if the user were to set, for example, linspace(0, 0.15, 100) therefore with steps of longer duration,
the algorithm should find about 60 steps for the shortest episode.  
If instead the user sets tlist = np.linspace(0, 0.1, 130) I think more steps are needed, the algorithm will find approximately the 100 minimum steps.  
The downside is that we will have many steps and therefore many actions, i.e. many values ​​of optimized_parameters even if they are similar to each other.  

The code for the second approach is the one above.   
The first approach can also be implemented by changing the time of mesolve() to max_episode_time and removing the last penalty term in the reward function.  
