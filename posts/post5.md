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

However, the procedure just seen works if I already have all the episodes that should be generated according to a fixed (non-optimal) policy. But afterwards we would also like to optimize the policy.  
To then be able to update the policy, an **Update rule** for V (and similar for Q) is more useful:  
$V(s_t) \leftarrow V(s_t) + \alpha(G_t - V(s_t))$ where $\alpha$ is the learning rate.  
which can be executed at the end of each episode, without having to have them all in advance.  
This will allow alternating the two phases of policy evaluation and improvement.  

Additionally, as the image suggests, there are two approaches to observing the rewards:  
**Every-visit MC** considers the state and reward every time it is visited in the episode.  
**First-visit MC** considers the state and reward only the first time it is visited in the episode.  

By definition $Q_\pi(s,a) \sim Avg[G_t|S_t=s,A_t=a]$  
Starting from a state $s$, we should consider an action $a$ that leads to a certain "branch"(red in the image below) that will have a return G in the episode, we should therefore average these returns of this branch for the various episodes. But this must be done for every action starting from $s$ and for every state $s$.  
This means that, in theory, one should explore all possible actions for each state (all pairs (s,a)) to ensure having all possible episodes and then find the optimal policy, which would take too much time.  
To ensure **exploration** in less time, one could use the **Exploring Starts** method, which consists of setting the initial (s,a) pair randomly for each episode.  
However, it is not always possible to randomly set this initial pair, which is why it is useful to adopt the ε-Soft Policy(see below).  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/2db56d59-672f-4d31-9ac2-254ec75145ba" alt="image" width="450"/>  
In practice, as mentioned before, the **Update rule** is used:  
$Q(s,a) \leftarrow Q(s,a) + \alpha(G_t - Q(s,a))$  
To be updated every time there is a new return $G_t$ for the pair $(s,a)$.  

## Policy Improvement
In DP, we saw the formula  
$\pi'(s) = \arg \max_a (r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_\pi(s'))$  
However, here we do not have $r(s,a)$ and the one-step dynamic $p(s'|s,a)$.  
We could simply consider the best action at each step with the following formula:  
$\pi'(s) = \arg \max_a Q_\pi(s,a)$  
But as said before, we should have all the pairs of $(s,a)$ to have a good result, but it would become infeasible...  
So to guarantee a good result you can use the following: 

**ε-Soft Policy**
In particular, we use an **ε-greedy policy**, which selects each action with a probability of at least $\frac{\epsilon}{|A|}$.  
Also as you progress you can decrease the value of $\epsilon$.  
$\pi'(s) = \arg \max_a Q_\pi(s,a)$ with $\frac{\epsilon}{|A|} + 1 - \epsilon$ probability  
Otherwise $\pi'(s) =$ another action, with probability $\frac{\epsilon}{|A|}$  
The more $\epsilon \to 1$, the more exploration is done (i.e., selecting actions that, according to current knowledge, are not the best but could be, which is the exploration-exploitation dilemma).

**Policy Iteration**: 
By alternating between policy evaluation (update rule) and policy improvement, you will converge to the optimal policy.  
