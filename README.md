# Research background
CRSA communication framework in multi-agent bandit setting
Game is negotiation and collaboration - need to give up on the best action for the self (conflicting private interest) to reach a consensus
Agent has its own preference but is not aware of the preference of the opponent
Agents reach the consensus by doing pragmatic reasoning through meanings and history of conversation, forming beliefs on the opponent’s private preference and reason
about each other’s intentions. 


examine the conversation history and the evolution of agents’ beliefs

# The matrix game
two agents are shown their own payoff matrix. However, they cannot see the opponent’s payoff matrix.
Our agents’ goal is to negotiate and reach an optimal joint action that maximizes the reward function, in our case, utilitarian (the max of sum)
In each turn within one negotiation, an agent will propose a joint action that is possible for him from the ensemble of all possible joint actions

In the next turn, the other agent can accept the proposal or reject and propose a new joint action.

This process will continue and agents will switch roles of listener and speaker in each turn until they reach an agreement. 

## Lexicon


## Vectorizarion
Notice that the code in CRSA module does not follow the exact steps of calculation as in the article. This is because the priors and lexicon
render a lot of terms 0. Thus, we take advantage of this and rewrite the equations to save computational complexity.
Vectorization is heavily used with this in order to achieve an acceptable computing time.

## Limitations
Meaning space

As of now, we limit the experiment to only 3x3 matrices. Until we come up with more abstraction on the meaning space,
augmenting to larger matrices will not only make the calculations impossible, but the memory simply will not allow it.

