Machine Learning (ML) is an alternative to traditional programming where a program is designed to learn from data, extracting relevant information or patterns autonomously, without explicit instructions.  

There are three main paradigms:
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
(img)  

### Return
What matters to the agent is not just the immediate reward but the long-term reward.  
We define the Return, starting from state $s_t$, as $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{k-1} R_{t+k}$  
$0 \le \gamma \le 1$ is the **Discount Factor**  
If $\gamma = 0$, only the immediate reward ($R_{t+1}$) matters.  
If $\gamma \to 1$, the problem becomes more complex because future rewards also matter.  
With $\gamma = 1$, it is a simple sum of future rewards.  
Note that $\gamma$ is not a hyperparameter to be "tuned"; instead, it depends on the type of problem at hand.  

(continued soon...)
