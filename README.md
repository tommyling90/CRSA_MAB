# CRSA in Multi-Agent Matrix Games

This repository implements and evaluates Communicative Rational Speech Act (CRSA) in a two-agent matrix-game negotiation setting, together with two deterministic greedy baselines.

The project studies how agents with **private and potentially conflicting preferences** can reach agreement through communication. Each agent observes only its own payoff matrix and does not know the preferences of the other agent. CRSA agents use pragmatic reasoning over the dialogue history to form beliefs about the opponent's private preferences and select proposals accordingly.

The implementation supports repeated experiments, comparison against greedy negotiation strategies, listener-prediction evaluation, and analysis of the agents' evolving beliefs.

---

## Game

Two agents, $A$ and $B$, are given private payoff matrices

$$
R_A,;R_B \in \mathbb{R}^{k\times k}.
$$

Each cell represents a joint action $y \in Y$. Agent $A$ observes $R_A$, while agent $B$ observes $R_B$.

The raw payoff matrices are transformed into private **meaning vectors**.

$$
m_A^{*},;m_B^{*},
$$

where actions are grouped into $n$ preference ranks. Rank $1$ represents the most preferred actions.

Each agent also has an acceptance threshold,

$$
\tau_A,\tau_B,
$$

such that an action is acceptable to agent $i$ when

$$
m_i(y)\leq\tau_i.
$$

A solution therefore exists when there is at least one joint action acceptable to both agents:

$$
\exists y\in Y:
m_A(y)\leq\tau_A
\land
m_B(y)\leq\tau_B.
$$

---

## Negotiation Protocol

Negotiation proceeds over a finite number of alternating turns.

At each turn, one agent acts as the **speaker** and proposes a joint action. The other acts as the **listener**. Their roles alternate after every turn.

There is no explicit accept message. A proposal is accepted by repeating the previous proposal. Therefore, agreement occurs when

$$
u_t=u_{t-1}.
$$

If no agreement is reached before the maximum number of turns, the episode terminates without agreement.

---

## Optimal Joint Action

The target joint action is defined using the agents' preference ranks.

For each joint action, consider the worse of the two agents' ranks:

$$
\max{m_A(y),m_B(y)}.
$$

The set of optimal actions is

$$
Y^* = \arg\min_{y\in Y}
\max{m_A(y),m_B(y)}.
$$

This is a **minimax-rank criterion**: it selects the action whose worst individual rank is as good as possible.

When $\tau_A=\tau_B=\tau$, the existence condition can equivalently be written as

$$
\min_{y\in Y}
\max{m_A(y),m_B(y)}
\leq\tau.
$$

Consequently, whenever a solution exists, the minimax-rank criterion returns a jointly acceptable solution.

Note that this differs from maximizing

$$
R_A(y)+R_B(y),
$$

which can select an action that strongly benefits one agent but lies outside the other agent's acceptance threshold.

Multiple actions may tie under the minimax criterion; therefore $Y^*$ is represented as a set/list of optimal joint actions throughout the implementation.

---

## CRSA

CRSA models communication as recursive pragmatic reasoning.

Rather than reasoning only from its own preference ordering, an agent maintains beliefs over the possible private meanings of its opponent. These beliefs are updated using the history of proposals and the predicted behavior of increasingly pragmatic speakers and listeners.

The recursion depth can be configured experimentally.

### Beliefs

Because agents cannot observe each other's payoff matrices or meanings, each agent maintains a distribution over possible opponent meanings.

After observing the opponent's utterances, this distribution is updated according to the probability that each possible opponent meaning would have generated the observed behavior.

This makes it possible to inspect how communication changes an agent's beliefs about its opponent over the course of negotiation.

### Lexicon

The lexicon represents the compatibility between utterances and meanings. An action is communicatively available only when it satisfies the relevant ranking constraints.

Conversation history is additionally used to restrict proposals while preserving the possibility of repeating the immediately preceding proposal for acceptance.

### Vectorization and Caching

A direct implementation of the recursive CRSA equations is computationally expensive because inference ranges over large meaning spaces.

The implementation therefore makes extensive use of NumPy vectorization and caching. Terms that are known to be zero from the prior, compatibility constraints, lexicon, or history are eliminated whenever possible rather than evaluated through the full recursive equations.

As a result, the implementation is algebraically optimized and does not always mirror the equations of the theoretical model line-by-line.

---

## Greedy Baselines

Two non-pragmatic negotiation strategies are provided for comparison with CRSA.

### Greedy I

Each agent proposes its highest-ranked remaining acceptable action.

When the opponent proposes an action satisfying

$$
m_i(y)\leq\tau_i,
$$

the agent accepts immediately by repeating the proposal.

Greedy I performs no inference about the opponent's private preferences.

### Greedy II

Greedy II is more selective.

Instead of immediately accepting every acceptable proposal, an agent continues proposing higher-ranked remaining actions while such alternatives are available. Once these preferred alternatives have been exhausted, the agent may accept a threshold-ranked proposal.

Actions ranked worse than the acceptance threshold are never proposed or accepted.

Both greedy algorithms are deterministic.

---

## Evaluation

Experiments can be run over multiple episodes and algorithms.

The implementation records metrics including:

* **agreement** — whether the agents reached any consensus;
* **optimal agreement** — whether the final agreed action belongs to $Y^*$;
* **turns to agreement**;
* **listener accuracy** across negotiation turns;
* CRSA belief distributions and their evolution.

### Listener Accuracy

At each relevant turn, the pragmatic listener produces a distribution over possible joint actions.

The highest-probability listener prediction is compared against the optimal-action set $Y^*$. When several actions tie for the highest listener probability, the prediction is counted as correct if at least one of them belongs to $Y^*$.

Thus,

$$
\mathrm{correct}_t =
\mathbf{1}
\left[
(\arg\max_y P_L(y\mid u,w,m_L)
\cap Y^*)
\neq\emptyset
\right].
$$

Listener accuracy can then be aggregated across episodes for each negotiation turn.

---

## Repository Structure

```text
CRSA_MAB/
├── configs/                 # Matrix and experiment YAML configurations
├── scripts/
│   └── run_experiment.py    # Experiment entry point
├── src/
│   ├── agents/              # Agent definitions
│   ├── algos/               # CRSA, Greedy I, and Greedy II
│   ├── envs/                # Matrix game and negotiation protocol
│   ├── priors/              # Priors and lexicon
│   ├── rewards/             # Optimality/reward functions
│   └── transforms/          # Matrix → meaning/space transformations
├── utils/                   # Plotting and utility functions
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tommyling90/CRSA_MAB.git
cd CRSA_MAB
```

Create and activate a virtual environment if desired, then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Run experiments from the repository root.

For example:

```bash
python scripts/run_experiment.py \
    --algorithm crsa \
    --matrix 3x3 \
    --episodes 10 \
    --reward-type minimax \
    --seed 42 \
    --recursion-depth 2 \
    --turns 9
```

Run Greedy I:

```bash
python scripts/run_experiment.py \
    --algorithm greedy \
    --matrix 3x3 \
    --episodes 10 \
    --seed 50 \
    --turns 9 \
    --reward-type minimax
```

Run Greedy II:

```bash
python scripts/run_experiment.py \
    --algorithm greedy_ii \
    --matrix 3x3 \
    --episodes 10 \
    --seed 50 \
    --turns 9 \
    --reward-type minimax
```

Matrix and parameter configurations can be supplied either by configuration name or YAML path. Command-line parameters override values specified in the parameter configuration.

To see all available arguments:

```bash
python scripts/run_experiment.py --help
```

---

## Computational Limitations

The principal computational bottleneck is the size of the meaning space.

For a (k\times k) game with (n) possible ranks, the unrestricted meaning space contains

$$
n^{k^2}
$$

possible meanings for each agent.

This grows extremely quickly. For example,

$$
3^{16}=43{,}046{,}721
$$

possible meanings are already required for a $4\times4$ game with three ranks, before accounting for the recursive CRSA computations.

The current implementation therefore focuses on relatively small matrix games. Vectorization, compatibility masks, and caching reduce computation substantially but do not eliminate the exponential growth of the underlying meaning space.

---

## Research Status

This repository is a research implementation and the research is under active development.