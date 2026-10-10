import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from payoffnash import NESeeking, QuadraticGame, ZOGrandientPlay, generate_quadratic_games


class TestSpec04Refactoring(unittest.TestCase):
    def setUp(self):
        games = generate_quadratic_games(ensemble_size=1, N=2, m=1, alpha=0.05, seed=42)
        self.game = games[0]
        self.initial_actions = np.array([[0.0], [0.0]])

        self.zo_estimator = ZOGrandientPlay(self.game, v=5.0, a=0.8, epsilon=1e-4)
        self.ne_estimator = NESeeking(
            game=self.game,
            k=np.array([[5.0], [5.0]]),
            a=np.array([[0.15], [0.15]]),
            omega=np.array([[25.0], [31.0]]),
            dt=0.005,
        )

    def test_missing_stopping_criterion_rejection(self):
        with self.assertRaises(ValueError):
            self.zo_estimator.estimate(self.initial_actions)

        with self.assertRaises(ValueError):
            self.ne_estimator.estimate(self.initial_actions)

    def test_history_length_and_initial_condition_zo(self):
        steps = 50
        history = self.zo_estimator.estimate(self.initial_actions, num_steps=steps)

        self.assertEqual(history.state_history.shape, (steps + 1, 2))
        self.assertEqual(history.action_history.shape, (steps + 1, 2))
        self.assertEqual(history.payoff_history.shape, (steps + 1, 2))
        np.testing.assert_allclose(history.state_history[0], self.initial_actions.ravel())
        np.testing.assert_allclose(history.action_history[0], self.initial_actions.ravel())

    def test_history_length_and_initial_condition_ne(self):
        steps = 50
        history = self.ne_estimator.estimate(self.initial_actions, num_steps=steps)

        self.assertEqual(history.state_history.shape, (steps + 1, 2))
        self.assertEqual(history.action_history.shape, (steps + 1, 2))
        self.assertEqual(history.payoff_history.shape, (steps + 1, 2))
        np.testing.assert_allclose(history.state_history[0], self.initial_actions.ravel())

    def test_time_limit_termination_zo(self):
        history = self.zo_estimator.estimate(
            self.initial_actions, num_steps=10_000_000, time_limit=0.05
        )
        self.assertLess(len(history.state_history), 10_000_000)

    def test_time_limit_termination_ne(self):
        history = self.ne_estimator.estimate(
            self.initial_actions, num_steps=10_000_000, time_limit=0.05
        )
        self.assertLess(len(history.state_history), 10_000_000)

    def test_max_rel_error_termination_zo(self):
        history = self.zo_estimator.estimate(
            self.initial_actions, num_steps=100_000, max_rel_error=10.0
        )
        self.assertLess(len(history.state_history), 100_000)

    def test_max_rel_error_termination_ne(self):
        omega_min = float(np.min(self.ne_estimator.omega))
        n_period = max(1, int(round((2.0 * np.pi) / (omega_min * self.ne_estimator.dt))))
        window_size = 5 * n_period

        history = self.ne_estimator.estimate(
            self.initial_actions, num_steps=100_000, max_rel_error=10.0
        )
        self.assertLess(len(history.state_history), 100_000)
        self.assertGreaterEqual(len(history.state_history), 2 * window_size)


if __name__ == "__main__":
    unittest.main()
