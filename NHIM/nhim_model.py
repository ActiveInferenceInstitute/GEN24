"""Reference implementation of the NHIM inquiry-agent model.

This module implements the discrete generative model specified in
``NHIM/nhim_inquiry_agent.gnn.md`` (Generalized Notation Notation, GNN v1).
The specification is the source of truth for the model's structure; the
matrices below are the canonical values declared in that file, and
``scripts/validate_gnn_spec.py`` checks the two against each other by parsing
the spec with the pinned upstream GNN parser.

Model (flat discrete POMDP):

- hidden state ``s`` — the inquiry context, 3 states:
  ``grounded`` / ``uncertain`` / ``conflicted``
- observation ``o`` — sense data, 3 outcomes:
  ``aligned`` / ``mixed`` / ``contradictory``
- action ``u`` — response mode, 3 actions: ``attune`` / ``reframe`` / ``defer``
- ``A`` likelihood ``P(o | s)``, ``B`` transition ``P(s' | s, u)`` (stored as
  ``B[next_state, previous_state, action]``), ``C`` log-preferences over
  observations, ``D`` uniform prior over states.

Distress is an auxiliary bounded scalar driven by belief uncertainty; it does
not feed back into the filter. Memory (Hopfield) and LLM response generation
in ``NHIM_API.py`` are integration components and are not part of this model.

Only ``numpy`` is required.
"""

import numpy as np

NUM_STATES = 3
NUM_OBSERVATIONS = 3
NUM_ACTIONS = 3

STATE_LABELS = ("grounded", "uncertain", "conflicted")
OBSERVATION_LABELS = ("aligned", "mixed", "contradictory")
ACTION_LABELS = ("attune", "reframe", "defer")

# A[observation, hidden_state] = P(o | s). Asymmetric sensor: each state most
# often produces its matching sense datum; columns are stochastic.
A = np.array(
    [
        [0.70, 0.15, 0.10],
        [0.20, 0.70, 0.25],
        [0.10, 0.15, 0.65],
    ],
    dtype=float,
)

# B[next_state, previous_state, action] = P(s' | s, u). Per-action slices are
# column-stochastic (rows are next states, columns are previous states):
#   action 0 "attune"  — hold the current context (identity)
#   action 1 "reframe" — keep the context with probability 0.7, else move to
#                        the grounded state
#   action 2 "defer"   — keep the context with probability 0.8, else spread
#                        uniformly over the three states
_B_PER_ACTION = np.array(
    [
        # action 0 "attune": identity
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        # action 1 "reframe": 0.7 identity, else toward grounded
        [[1.0, 0.3, 0.3], [0.0, 0.7, 0.0], [0.0, 0.0, 0.7]],
        # action 2 "defer": 0.8 identity, else uniform
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
    ],
    dtype=float,
)
B = np.transpose(_B_PER_ACTION, (1, 2, 0)).copy()

# C[observation]: log-preferences. Prefer aligned sense data, avoid
# contradictory sense data.
C = np.array([1.0, 0.0, -1.0], dtype=float)

# D[state]: uniform prior over the inquiry context.
D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)

DISTRESS_DECAY = 0.1
DISTRESS_MAX = 5.0
RELIEF_INTERVAL = 5
RELIEF_FACTOR = 0.5


def initial_beliefs():
    """Return the prior over hidden states (a copy of ``D``)."""
    return D.copy()


def predict(state_beliefs, action):
    """Return ``B[:, :, action] @ state_beliefs`` — the next-state prior."""
    return B[:, :, action] @ np.asarray(state_beliefs, dtype=float)


def likelihood(observation):
    """Return ``A[observation, :]`` — the likelihood of one sense datum."""
    return A[observation, :]


def update_beliefs(state_beliefs, observation, action):
    """Exact filtering step: posterior over the next state given the datum.

    ``q_t(s) ∝ A[o_t, s] · (B[:, :, u] q_{t-1})(s)``. The result sums to 1.
    """
    prior = predict(state_beliefs, action)
    posterior = prior * likelihood(observation)
    total = posterior.sum()
    if total <= 0:
        raise ValueError("Observation has zero probability under the model")
    return posterior / total


def variational_free_energy(state_beliefs, prior, observation):
    """Return the variational free energy of one belief update."""
    q = np.asarray(state_beliefs, dtype=float)
    r = np.asarray(prior, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        complexity = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0) / r), 0.0).sum()
        accuracy = (q * np.log(likelihood(observation))).sum()
    return float(complexity - accuracy)


def predicted_observations(state_beliefs, action):
    """Return ``P(o | u)`` for one action, from the current beliefs."""
    return A @ predict(state_beliefs, action)


def expected_free_energy(state_beliefs, action):
    """Return the expected free energy ``G(u)`` for one action.

    ``G(u) = Σ_o P(o|u) [log P(o|u) − C[o]]`` — expected ambiguity minus
    expected log-preference. Lower is better.
    """
    p_o = predicted_observations(state_beliefs, action)
    with np.errstate(divide="ignore", invalid="ignore"):
        ambiguity = np.where(p_o > 0, p_o * np.log(np.where(p_o > 0, p_o, 1.0)), 0.0).sum()
    risk = (p_o * C).sum()
    return float(ambiguity - risk)


def expected_free_energies(state_beliefs):
    """Return ``G(u)`` for every action, in action order."""
    return np.array(
        [expected_free_energy(state_beliefs, action) for action in range(NUM_ACTIONS)],
        dtype=float,
    )


def policy_posterior(state_beliefs, precision=1.0):
    """Return the softmax policy posterior ``π(u) ∝ exp(−precision · G(u))``."""
    g = expected_free_energies(state_beliefs)
    logits = -precision * (g - g.min())
    weights = np.exp(logits)
    return weights / weights.sum()


def select_action(state_beliefs):
    """Return the action with the lowest expected free energy."""
    return int(np.argmin(expected_free_energies(state_beliefs)))


def update_distress(distress, state_beliefs, step=0):
    """Update the bounded distress scalar from belief uncertainty.

    ``d' = clip((1 − decay) · d + (1 − max_s q(s)), 0, max)``, with a periodic
    relief (``d ← 0.5 · d``) every ``RELIEF_INTERVAL`` steps.
    """
    confidence = float(np.max(np.asarray(state_beliefs, dtype=float)))
    distress = (1.0 - DISTRESS_DECAY) * float(distress) + (1.0 - confidence)
    if step > 0 and step % RELIEF_INTERVAL == 0:
        distress *= RELIEF_FACTOR
    return float(np.clip(distress, 0.0, DISTRESS_MAX))
