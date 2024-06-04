# Monte Carlo (MC)
In the previously discussed Value Iteration formula  
$V_{k+1}(s) \leftarrow \max\limits_a [r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_k (s')], \forall s$
the major limitation is that typically we do not know the dynamics of the problem $p(s'|s,a)$ and the reward $r(s,a)$.  
We want something that learns the optimal policy directly from the data!

The disadvantage is that it can be a slow method (because it needs to learn from data) and it requires episodic tasks, meaning it must always reach a terminal state.  

Here, too, the policy iteration method of two steps is executed: policy evaluation and policy improvement.

## Policy Evaluation
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/5958e17e-c552-467b-8599-db5691339506" alt="image" width="450"/>  

We need to find $V_\pi(s)$ and we have data (episodes) represented in the image as squares stacked on top of each other.  
By definition, $V_\pi(s) = E_\pi[G_t|S_t=s]$  
The idea is to average the returns $G_t$ obtained starting from a state (the average is for the number of episodes we have, not fot the number of states).  
$V_\pi(s) \sim Avg[G_t|S_t=s]$  

One should then calculate all the return $G_t$ of the episodes and then take the average.  
Computationally, it is more useful to use the following **Update Rule**:
$V(s_t) \leftarrow V(s_t) + \alpha(G_t - V(s_t))$  
where $\alpha$ is the learning rate.  

Additionally, as the image suggests, there are two approaches to observing the rewards:  
**Every-visit MC** considers the state and reward every time it is visited in the episode.  
**First-visit MC** considers the state and reward only the first time it is visited in the episode.  

## Policy Improvement
In DP, we saw the formula  
$\pi'(s) = \arg \max_a (r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_\pi(s'))$  
However, here we do not have $r(s,a)$ and the one-step dynamic $p(s'|s,a)$.  
We use the Action value function $\pi'(s) = \arg \max_a Q_\pi(s,a)$,  
but $Q_\pi(s,a) \sim Avg[G_t|S_t=s,A_t=a]$  
This means that, in theory, one should explore all possible actions for each state (all pairs (s,a)) to ensure having all possible episodes and then find the optimal policy, which would take too much time.  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/2db56d59-672f-4d31-9ac2-254ec75145ba" alt="image" width="450"/>  
To ensure **exploration** in less time, one could use the **Exploring Starts** method, which consists of setting the initial (s,a) pair randomly for each episode.  
However, it is not always possible to randomly set this initial pair, which is why it is useful to adopt the following policy.  

### ε-Soft Policy
In particular, we use an **ε-greedy policy**, which selects each action with a probability of at least $\frac{\epsilon}{|A|}$.  
$\pi(s|a) = \frac{\epsilon}{|A|} + 1 - \epsilon ,$ if $a=a^\*$. Otherwise $\pi(s|a) = \frac{\epsilon}{|A|}$  
The more $\epsilon \to 1$, the more exploration is done (i.e., selecting actions that, according to current knowledge, are not the best but could be, which is the exploration-exploitation dilemma).
