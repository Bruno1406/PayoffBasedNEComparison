import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from payoffnash.common import Game
from payoffnash.ne_seeking import NESeeking5D
from payoffnash.zo_gradient_play import ZOGrandientPlay


@pytest.fixture
def simple_game():
    # Simple decoupled quadratic game where NE is [0.5, 0.5]
    def p1(a1, a2):
        return -((a1 - 0.5) ** 2)

    def p2(a1, a2):
        return -((a2 - 0.5) ** 2)

    return Game(
        N=2,
        action_sets=[[(-1.0, 1.0)], [(-1.0, 1.0)]],
        payoff_functions=[p1, p2],
    )


def test_early_stopping_on_true_ne_zo(simple_game):
    zo = ZOGrandientPlay(simple_game, v=3.0, a=0.5, epsilon=1e-5)
    true_ne = np.array([0.5, 0.5])

    # Initial state is far: [0.0, 0.0]
    hist = zo.estimate(
        initial_actions=np.array([0.0, 0.0]),
        num_steps=10000,
        true_ne=true_ne,
        ne_tol=0.2,
    )

    final_state = hist.state_history[-1]
    final_dist = np.linalg.norm(final_state - true_ne)
    assert final_dist <= 0.2
    assert len(hist.action_history) < 10000


def test_early_stopping_on_true_ne_ne_seeking(simple_game):
    ne = NESeeking5D(
        simple_game,
        k=np.array([[1.0], [1.0]]),
        a=np.array([[0.1], [0.1]]),
        omega1=10.0,
        dt=0.01,
    )
    true_ne = np.array([0.5, 0.5])

    hist = ne.estimate(
        initial_u_hat=np.array([0.4, 0.4]),
        num_steps=10000,
        true_ne=true_ne,
        ne_tol=0.2,
    )

    final_state = hist.state_history[-1]
    final_dist = np.linalg.norm(final_state - true_ne)
    assert final_dist <= 0.2
    assert len(hist.action_history) < 10000


def test_no_stopping_criterion_raises_error(simple_game):
    zo = ZOGrandientPlay(simple_game)
    with pytest.raises(ValueError, match="At least one stopping criterion"):
        zo.estimate(initial_actions=np.array([0.0, 0.0]))

    ne = NESeeking5D(
        simple_game,
        k=np.array([[1.0], [1.0]]),
        a=np.array([[0.1], [0.1]]),
        omega1=10.0,
        dt=0.01,
    )
    with pytest.raises(ValueError, match="At least one stopping criterion"):
        ne.estimate(initial_u_hat=np.array([0.0, 0.0]))
