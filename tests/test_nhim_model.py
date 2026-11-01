"""Tests for the GNN-specified NHIM inquiry model (NHIM/nhim_model.py)."""

import numpy as np
import pytest

import nhim_model as m


def test_matrices_have_declared_dimensions():
    assert m.A.shape == (3, 3)
    assert m.B.shape == (3, 3, 3)
    assert m.C.shape == (3,)
    assert m.D.shape == (3,)


def test_likelihood_and_transition_are_stochastic():
    assert np.allclose(m.A.sum(axis=0), np.ones(3))
    for action in range(m.NUM_ACTIONS):
        assert np.allclose(m.B[:, :, action].sum(axis=0), np.ones(3))


def test_transition_slices_match_their_documented_policies():
    assert np.allclose(m.B[:, :, 0], np.eye(3))  # attune: identity
    assert np.allclose(m.B[:, :, 1][:, 0], [1.0, 0.0, 0.0])  # reframe
    assert np.allclose(m.B[:, :, 2][:, 0], [0.8, 0.1, 0.1])  # defer


def test_update_beliefs_is_normalized_and_matches_one_manual_step():
    posterior = m.update_beliefs(m.initial_beliefs(), observation=0, action=0)
    assert posterior.sum() == pytest.approx(1.0)
    expected = m.A[0, :] * m.D
    assert np.allclose(posterior, expected / expected.sum())


def test_update_beliefs_follows_the_transition_prior():
    # After "reframe" from the conflicted state, most mass moves to grounded.
    q = np.array([0.0, 0.0, 1.0])
    prior = m.predict(q, action=1)
    assert prior[0] == pytest.approx(0.3)
    assert prior[2] == pytest.approx(0.7)


def test_update_beliefs_rejects_impossible_observations(monkeypatch):
    broken = m.A.copy()
    broken[0, :] = 0.0  # observation 0 becomes impossible under every state
    monkeypatch.setattr(m, "A", broken)
    with pytest.raises(ValueError):
        m.update_beliefs(m.initial_beliefs(), observation=0, action=0)


def test_variational_free_energy_at_the_posterior_equals_surprise():
    prior = m.predict(m.initial_beliefs(), action=0)
    observation = 1
    posterior = m.update_beliefs(m.initial_beliefs(), observation, action=0)
    surprise = -np.log(float(prior @ m.A[observation, :]))
    assert m.variational_free_energy(posterior, prior, observation) == pytest.approx(
        surprise, abs=1e-9
    )


def test_expected_free_energy_selects_the_minimum():
    q = m.initial_beliefs()
    values = m.expected_free_energies(q)
    assert values.shape == (3,)
    assert np.isfinite(values).all()
    assert m.select_action(q) == int(np.argmin(values))


def test_policy_posterior_is_a_distribution():
    posterior = m.policy_posterior(m.initial_beliefs())
    assert posterior.sum() == pytest.approx(1.0)
    assert (posterior > 0).all()


def test_preferences_make_the_grounded_state_attractive():
    # From the conflicted state, the model should not prefer "attune".
    q = np.array([0.0, 0.0, 1.0])
    assert m.select_action(q) != 0


def test_distress_is_bounded_and_relieved_periodically():
    distress = 0.0
    beliefs = m.initial_beliefs()
    for step in range(1, 30):
        distress = m.update_distress(distress, beliefs, step=step)
        assert 0.0 <= distress <= m.DISTRESS_MAX
    # The relief step halves the level relative to the same update without it.
    without_relief = (1 - m.DISTRESS_DECAY) * 4.0 + (1 - 1 / 3)
    assert m.update_distress(4.0, np.array([1.0, 0.0, 0.0]), step=5) < without_relief


def test_updates_are_deterministic():
    first = m.update_beliefs(m.initial_beliefs(), observation=2, action=1)
    second = m.update_beliefs(m.initial_beliefs(), observation=2, action=1)
    assert np.array_equal(first, second)
