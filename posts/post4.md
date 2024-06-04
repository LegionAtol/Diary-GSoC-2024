# Dynamic Programming (DP)
As seen in the previous post, we need to find the optimal policy.
The idea of DP is, as we will see, to iterate two steps:  

- Policy Evaluation, to evaluate a given policy  
- Policy Improvement, to improve the policy  

## Policy Evaluation
Observing the Bellman equation already seen: $V_\pi(s) = \sum\limits_a \pi(s|a) [r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_\pi (s')]$
the idea is to use it iteratively, obtaining the following **Update Rule**:  
$V_{k+1}(s) \leftarrow \sum\limits_a \pi(s|a) [r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_k (s')]$  
At each iteration k, $V_k$ is updated for all states $s$.  
$V_0 \rightarrow V_1 \rightarrow V_2 \rightarrow V_3 \rightarrow \dots \rightarrow V_k \rightarrow \dots \rightarrow V_\pi$  
As $k \to \infty$, $V_k$ converges to $V_\pi$.  
Note that it does not converge to the optimal policy; this is only the policy evaluation phase, and we are not modifying the given policy.  
With this Update Rule, we have solved the problem seen in the previous post of obtaining large linear systems to evaluate the Value function.  

## Policy Improvement
As seen in the previous post, to find the optimal policy $\pi^\*$ or its $V_{\pi^\*}$, we obtained a system in which the maximum appeared.  
We found:  
$\pi^*(s) = \arg \max_a Q^\*(s,a) = \arg \max_a (r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V^\*(s'))$  
Since we don't have $Q^\*$ and $V^\*$, the idea here is to act **greedy** with respect to a non-optimal value function.  
$\pi'(s) = \arg \max_a Q_\pi(s,a) = \arg \max_a (r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_\pi(s')), \forall s$  
By doing so, we have two cases: either the policy $\pi'$ is already the optimal policy, or $\pi'$ will be better than $\pi$. (this can be proven)  

## Policy Iteration
As mentioned at the beginning, the idea is to iterate the two steps of Policy Evaluation and Policy Improvement to arrive at the optimal policy and the optimal value function.  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/64e06926-58db-438f-a028-6ff94f7a18fd" alt="image" width="550"/>  
As you can imagine, the main problem with this method is that it is very slow because the two steps must be executed every time over the entire state space.  
This is where the following method can be useful.  

## Generalized Policy Iteration (GPI)
**Value Iteration** is the most popular GPI.  
The idea is to combine the two steps of Policy Evaluation and Policy Improvement.  
Combining the two formulas just seen, we obtain:  
$V_{k+1}(s) \leftarrow \max\limits_a [r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_k (s')], \forall s$  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/8fd54384-52f5-449f-92d0-0f62d4607b0a" alt="image" width="300"/>  
It can be shown that $\lim_{k \to \infty} V_k=V^\*$  
The algorithm remains quite slow $O(|S^2||A|)$, but there are variants to speed it up.  
In practice, DP is only suitable for relatively small problems, with a few million states.
