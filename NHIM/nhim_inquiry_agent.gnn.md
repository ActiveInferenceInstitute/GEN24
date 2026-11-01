# GNN Specification: NHIM Inquiry Agent

# GNN Version: 1.0

# Canonical discrete model of the NHIM inquiry loop (NHIM/NHIM_API.py).
# Validated against the pinned upstream parser by scripts/validate_gnn_spec.py
# using NHIM/gnn-pin.json. The reference implementation of this model is
# NHIM/nhim_model.py; the two must declare the same matrices.

## GNNSection

NHIMInquiryAgent

## GNNVersionAndFlags

GNN v1

## ModelName

NHIM Inquiry Agent

## ModelAnnotation

The NHIM module in this repository sketches a problem-solving agent whose
inquiry loop carries a three-element belief vector, sense data, response
actions, and a bounded distress level. This file is the canonical discrete
generative model of that loop; `NHIM/nhim_model.py` implements exactly the
structure declared here.

- Hidden state `s` — the inquiry context, 3 states: `grounded`,
  `uncertain`, `conflicted` (the sketch's 3-element belief vector).
- Observation `o` — sense data, 3 outcomes: `aligned`, `mixed`,
  `contradictory` (the sketch's `P_d_given_s` sense-data categories).
- Action `u` — response mode, 3 actions: `attune`, `reframe`, `defer`
  (the sketch's three response actions).
- `A` is the likelihood `P(o | s)`, `B` the transition `P(s' | s, u)`
  (`B[next_state, previous_state, action]`), `C` log-preferences over sense
  data, `D` the uniform prior over contexts.
- Distress is an auxiliary bounded scalar (`distress`) driven by belief
  uncertainty; it does not feed back into the filter.
- The Hopfield memory and LLM response generation in `NHIM_API.py` are
  integration components, not part of this model.

Provenance: the sketch initializes `P_d_given_s`, `P_s_prime_given_s_a`,
`P_s`, and `v_s_s_prime` with `numpy.random.rand` (values are not
reproducible and the placeholders are never calibrated). This specification
replaces those random initializations with canonical, documented values:
an asymmetric sensor, three named response policies, a preference for
aligned sense data, and a uniform context prior. `NHIM/nhim_model.py` is the
single implementation of these values.

## StateSpaceBlock

# Generative model parameters

A[3,3,type=float]     # Likelihood: P(observation | hidden state)
B[3,3,3,type=float]   # Transition: B[next_state, previous_state, action]
C[3,type=float]       # Log-preferences over observations
D[3,1,type=float]     # Prior over hidden states

# Hidden state and inference quantities

s[3,1,type=float]       # Beliefs over the inquiry context
s_prime[3,1,type=float] # Predicted next-state beliefs
F[1,type=float]         # Variational free energy of a belief update

# Observation

o[3,1,type=int]       # Sense data (one-hot encoded)

# Policy and control

π[3,type=float]       # Policy posterior over the three response modes
G[3,type=float]       # Expected free energy per response mode
u[1,type=int]         # Selected response mode

# Auxiliary state

distress[1,type=float] # Bounded distress level in [0, 5]
t[1,type=int]          # Discrete time step

## Connections

D>s
s-A
A-o
s-B
B>s_prime
u>s_prime
s>G
C>G
G>π
π>u
s-F
o-F
s>distress
t>distress

## InitialParameterization

# A: rows are observations, columns are hidden states (grounded, uncertain,
# conflicted). An asymmetric sensor: each state most often produces its
# matching sense datum.
A={
  (0.70, 0.15, 0.10),
  (0.20, 0.70, 0.25),
  (0.10, 0.15, 0.65)
}

# B: stored as B[next_state, previous_state, action]; each per-action slice is
# column-stochastic (rows are next states, columns are previous states, each
# column sums to 1 over next states). Action 0 "attune" holds the context
# (identity); action 1 "reframe" holds it with probability 0.7 and otherwise
# moves to the grounded state; action 2 "defer" holds it with probability 0.8
# and otherwise spreads uniformly over the three states.
B={
  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),
  ( (1.0,0.3,0.3), (0.0,0.7,0.0), (0.0,0.0,0.7) ),
  ( (0.8,0.1,0.1), (0.1,0.8,0.1), (0.1,0.1,0.8) )
}

# C: log-preferences over sense data. Aligned sense data is preferred,
# contradictory sense data is avoided.
C={(1.0, 0.0, -1.0)}

# D: uniform prior over the three inquiry contexts.
D={(0.333333, 0.333333, 0.333333)}

## Equations

Perception is exact online filtering. With prior D at t = 0 and action u:

r_t(s) = sum_j B[s,j,u] q_{t-1}(j)
q_t(s) = A[o_t,s] r_t(s) / sum_j A[o_t,j] r_t(j)

The per-step variational free energy is
F_t(q) = sum_s q_t(s) log(q_t(s)/r_t(s)) - sum_s q_t(s) log A[o_t,s].

Policy selection uses expected free energy over the three response modes.
The predicted observation distribution for action u is
P(o|u) = sum_{s'} A[o,s'] (B[:,:,u] q_t)(s'), and
G(u) = sum_o P(o|u) [ log P(o|u) - C[o] ],
the expected ambiguity minus the expected log-preference; lower is better.
The policy posterior is π(u) proportional to exp(-G(u)), and the selected
response is u = argmin_u G(u) (the implementation reports both).

Distress is auxiliary and bounded by d_max = 5.0:
d_{t+1} = clip((1 - 0.1) d_t + (1 - max_s q_t(s)), 0, d_max),
with periodic relief d <- 0.5 d every 5 steps. Belief uncertainty therefore
raises distress; relief and decay lower it.

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=10

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
s=HiddenState
s_prime=NextHiddenState
o=Observation
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
π=PolicyVector
u=Action
t=Time

## ModelParameters

num_hidden_states: 3
num_obs: 3
num_actions: 3
num_timesteps: 10
distress_decay: 0.1
distress_max: 5.0
relief_interval: 5
relief_factor: 0.5

## Footer

NHIM Inquiry Agent v1 - GNN representation.
Canonical discrete model of the NHIM inquiry loop; reference implementation in
NHIM/nhim_model.py.

## Signature

Cryptographic signature goes here
