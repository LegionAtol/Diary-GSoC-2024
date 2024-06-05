# Temporal Difference (TD)
As seen, DP requires knowledge of MDP dynamics, while MC can be applied to episodic tasks and learns only after the episode is over, which can be limiting.  

## SARSA
It is one of the most famous **on-policy** TD algorithms.
On-policy means that the same policy is used both to collect experiences and to update the policy itself.  
Let's look at the two steps of policy evaluation and improvement.  

### Policy Evaluation
The update rule of the Value function is $V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$  
But to get the return $G_t$, you don't have to wait until the end of the episode, $G_t \sim R_{t+1} + \gamma V(S_{t+1})$, the return is the immediate reward + the discounted Value function of the next state.  
We obtain:  
$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$
It is known as TD(0)
The part in the square brackets is called **Temporal Difference error**  
Thus TD can learn at each step and does not have to wait until the end of the episode to learn.  

Compared to MC, TD provides a biased estimate of $V_\pi(S_t)$ since it uses $V(S_{t+1})$  
However, TD has lower variance.  

Similarly, we have:  
$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$  
with $Q_\pi(s,a)=Avg[G_t|S_t=s,A_t=a]$ and  
$G_t \sim R_{t+1} + \gamma Q(S_{t+1},A_{t+1})$  

### Policy Improvement
As seen in MC, we have $\pi'(s) = \arg \max_a Q_\pi(s,a)$ and the ε-greedy policy.  

## Q-Learning
It is a famous **off-policy** method, aiming to decouple the data collection process from policy training.  
This means there is a behavior policy used to collect experience, to perform the agent's actions. Often it is an ε-greedy policy for exploration.  
Then there is a target policy, which is greedy, used to update the Value function.  

### Policy Evaluation
The following update rule is used:  
$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(R_{t+1} + \gamma \max\limits_a Q(S_{t+1},a) - Q(S_t,A_t))$  
Note that Q does not have the subscript $\pi$ to highlight the off-policy nature. In practice, the next state and rewards found by the behavior policy are used.

### Policy Improvement
The action with the highest Q function value is chosen for all states:  
$\pi(s) = \arg\max_a Q(s,a)$  
