To better understand this theoretical part I suggest the following points:
- Have knowledge of mathematics (Mathematical Analysis 1)
- Have knowledge of probability and statistics
- Read the topics in sequence because I will use concepts and notations expressed previously
- Imagining simple and concrete examples where applying the formulas can help  

With that said, let's get started!

**Machine Learning** (ML) is an alternative to traditional programming where a program is designed to learn from data, extracting relevant information or patterns autonomously, without explicit instructions.  

There are three main **paradigms**:
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

## Supervised Learning
A model is trained on a labeled dataset, meaning each input has a desired output. The goal is to learn a function that maps inputs to outputs.  
Two typical tasks are Classification and Regression (for example, with the [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)).  
(img)

## Unsupervised Learning
Similar to the previous case but without labels, meaning no output for the various inputs. Here, the program must find patterns or structures within the data itself.  
Two common tasks are Clustering and Dimensionality Reduction.
(img)

## Reinforcement Learning
Different from the previous ones, RL is based on a try-error process. The agent interacts with the environment through actions (a), receives feedback through rewards (r), and ends up in a new state (s'), and so on. It is useful for problems where decisions must be made in sequence; essentially, the agent does not know the best action to take at each moment but has a long-term goal to achieve.  
(img agent and environment)

Here, I will delve deeper into RL as it seems to be the most interesting technique for my project. We will explore the theory behind the main techniques: Dynamic Programming(DP), Monte Carlo(MC), and Temporal DIfference(TD).  
I will also aim to provide an intuitive and practical understanding of the formulas.  
These theoretical foundations will be useful for better understanding (and being able to adapt/modify) the techniques used in various papers.  

Before starting with DP, let's review some key concepts and definitions in RL.  

The mantra to remember in RL is: "specify what you want to achieve, not how to achieve it"  
This is encapsulated in the **Reward Hypothesis**: in RL, the goal must be described as the maximization of cumulative rewards. Designing an effective reward function for the problem is not a simple task, and there are various approaches.  

### Markov Decision Problem (MDP)
The **Markov Property** holds: the future state (s') and reward (r) depend only on the current state and action (a).  
We will consider **Finite MDPs**, meaning the sets of actions ($A$) and states ($S$) are finite.

### One-step Dynamic
$p(s',r|s,a)$ is the function that describes the probability of transitioning to a state $s'$ and receiving a reward $r$ if the agent is in state $s$ and takes action $a$.  
Note that it is a probability, reflecting the stochastic nature of real-world situations, such as a robot performing actions in a difficult and not entirely "predictable" environment.  

$r(s,a)=E[R_{t+1}|S_t=s, A_t=a]$ is the expected value of the reward that can be obtained starting from state $s$ and taking action $a$.  

### Episodic Task
Episodic tasks represent the concept of agent-environment interaction as a series of states, actions, and rewards.  
A single instance defined by certain states, actions, and rewards that concludes in a terminal state is called an **Episode**.  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/ce7cdee9-8b6a-4911-ada1-3296f86dd0d0" alt="image" width="450"/>  

### Return
What matters to the agent is not just the immediate reward but the long-term reward.  
We define the Return, starting from state $s_t$, as $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{k-1} R_{t+k}$  
$0 \le \gamma \le 1$ is the **Discount Factor**  
If $\gamma = 0$, only the immediate reward ($R_{t+1}$) matters.  
If $\gamma \to 1$, the problem becomes more complex because future rewards also matter.  
With $\gamma = 1$, it is a simple sum of future rewards.  
Note that $\gamma$ is not a hyperparameter to be "tuned"; instead, it depends on the type of problem at hand.  

### Policy
The policy defines the actions the agent should take in various situations.  
A **Deterministic Policy** $\pi(s)=a$ is a function that describes with certainty, given a state, the action the agent will take. It can be described with a table of states and actions.  
More generally, a **Stochastic Policy** $\pi(s|a)$ is often used as a function that maps each state $s$ to the probability of executing a certain action $a$.  
It can also be used to represent a Deterministic Policy, which is a special case where the probability = 1.  

It is called a **Markovian Policy** if the Markov property holds; otherwise, it is called a **Non-Markovian Policy**.  

Generally, we consider **Stationary Policies**, meaning they do not change over time.  

Now let's look at two fundamental concepts in RL.  
### Value Function
Also known as the **State-Value Function**, it represents the expected reward that an agent can obtain starting from a state $s$ and following a given policy $\pi$.  
$V_\pi(s) = E_\pi [ G_t | S_t = s ] =
E_\pi [\sum\limits_{k=1}^{\infty} \gamma^k R_{t+k+1} | S_t=s]$  
As can be inferred from the definition, this function is useful for evaluating one policy against another.  

### Action Function
Also known as the **State-Action Function**, it represents the expected reward that an agent can obtain starting from a state $s$, performing an initial action $a$, and then following a certain policy $\pi$.  
$Q_\pi(s,a) = E_\pi [ G_t | S_t = s, A_t = a ] =
E_\pi [\sum\limits_{k=1}^{\infty} \gamma^k R_{t+k+1} | S_t=s, A_t=a]$  
The Action function is useful for comparing the effects of different actions in a state.  
(imagine playing chess and having to compare what the best action might be, given a certain state/configuration of the game)  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/419f85bf-6ebe-4828-b063-c93eeae6df25" alt="image" width="250"/>  

The $V$ and $Q$ functions are connected; the State-Value function can be obtained from the Action-Value function as follows:  
$V_\pi(s) = \sum\limits_a \pi(a|s) Q_\pi(s, a)$  
In practice, the $V$ function can be seen as a weighted sum of the Action-Value function of the possible actions in that state.  
The weighting is given by the probability with which the policy chooses each action in that state.

Suppose we have the following `example` with 4 states A, B, C, and D and a policy that moves randomly with a certain value of $\gamma$.   
The reward I get is 0 everywhere except for actions that take me to state B, where the reward is 5.  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/513bec28-549f-4ce9-a183-8bc17190246a" alt="image" width="250"/>  

If I try to calculate the Value function using the definition $V_\pi(s) = E_\pi [\sum\limits_{k=1}^{\infty} \gamma^k R_{t+k+1} | S_t=s]$ it wouldn't be very useful because I would have an infinite sum.  
I can rewrite the $V$ and $Q$ functions in the following way.  

### Bellman Equation
Observing that the Return $G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots =
R_t + \gamma(R_{t+1} + \gamma R_{t+2} + \dots)$  
The Value function can be decomposed into the immediate reward plus the discounted value of successor states.  
$V_\pi(s) = E_\pi[R_{t} + \gamma G_{t+1} | S_t=s]$  
To emphasize that $R_{t}$ is the reward obtained after performing the action, it is denoted as $R_{t+1}$  
$V_\pi(s) = E_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) | S_t=s]$  
Which can be rewritten as  
$V_\pi(s) = \sum\limits_a \pi(s|a) [r(s|a) + \gamma \sum\limits_{s'} p(s'|s,a) V_\pi (s')]$  
$s'$ denotes the next state.  

Similarly, for the Action-Value function, I get
$Q_\pi(s,a) = E_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) | S_t=s, A_t=a]$  
$= r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) V_\pi(s') =
r(s,a) + \gamma \sum\limits_{s'} p(s'|s,a) \sum\limits_{a'} \pi(a'|s') Q_\pi(s',a')$  
Where I have used the fact that $V$ and $Q$ are connected as seen.  
Note that both are **recursive** formulas.

Returning to the previous `example`, I can write the Value function for state A using the new recursive form  
$V_\pi(s) = \frac{1}{4}(5 + \gamma V_\pi(B)) + \frac{1}{4}(0 + \gamma V_\pi(C)) + \frac{1}{4}(0 + \gamma V_\pi(A) + \frac{1}{4}(0 + \gamma V_\pi(A)))$  
Similarly, I write the V function for the other states, $V(B), V(C), V(D)$.  
Thus, I obtain a **linear system** of 4 equations and 4 unknowns that is solvable.  

This was just a simple example, but real problems are much more complex, just think about increasing the number of states in the previous example. It would become **computationally infeasible**.  
We will see the solution to this problem shortly when we talk about DP and the update rule in policy evaluation.

### Optimal Policy
Now, let's see how to find an optimal policy, often denoted as $\pi^*$  
A policy $\pi$ is said to be greater than or equal to $\pi'$ if and only if $V_\pi(s) \ge V_{\pi'}(s), \forall(s)$  
<img src="https://github.com/LegionAtol/Diary-GSoC-2024/assets/118752873/dad5d1d7-3e27-4665-ae01-d66d1e460d20" alt="image" width="250"/>  


