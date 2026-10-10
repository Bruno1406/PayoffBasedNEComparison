import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from payoffnash import (
    Game,
    NESeeking,
    QuadraticGame,
    QuadraticGameGenerator,
    ZOGrandientPlay,
    generate_quadratic_games,
)


class TestGameEnsemble(unittest.TestCase):
    def test_generation_count_and_types(self):
        ensemble_size = 5
        games = generate_quadratic_games(ensemble_size=ensemble_size, N=2, m=1, seed=42)
        self.assertEqual(len(games), ensemble_size)

        for game in games:
            self.assertIsInstance(game, QuadraticGame)
            self.assertIsInstance(game, Game)
            self.assertEqual(game.N, 2)
            self.assertEqual(len(game.action_sets), 2)
            self.assertEqual(len(game.payoff_functions), 2)

            for action_set in game.action_sets:
                for lb, ub in action_set:
                    self.assertLess(lb, 0.0)
                    self.assertGreater(ub, 0.0)

    def test_strong_monotonicity_and_hurwitz(self):
        alpha = 1e-3
        games = generate_quadratic_games(ensemble_size=4, N=3, m=1, alpha=alpha, seed=123)
        for game in games:
            H = game.H
            H_sym = 0.5 * (H + H.T)
            max_eig_sym = np.linalg.eigvalsh(H_sym)[-1]
            self.assertLessEqual(max_eig_sym, -alpha + 1e-10)

            eigs_H = np.linalg.eigvals(H)
            self.assertTrue(np.all(np.real(eigs_H) < 0.0))

    def test_interior_nash_equilibrium_and_optimality(self):
        games = generate_quadratic_games(ensemble_size=3, N=2, m=1, enforce_interior=True, seed=99)
        for game in games:
            u_star = game.nash_equilibrium
            flat_bounds = [b for act in game.action_sets for b in act]

            for d, (lb, ub) in enumerate(flat_bounds):
                self.assertGreater(u_star[d], lb)
                self.assertLess(u_star[d], ub)

            u_list = [float(x) for x in u_star]
            eval_payoffs = [float(f(*u_list)) for f in game.payoff_functions]
            np.testing.assert_allclose(eval_payoffs, game.optimal_payoffs, atol=1e-10)

            eps = 1e-6
            for i in range(game.N):
                u_plus = list(u_list)
                u_minus = list(u_list)
                u_plus[i] += eps
                u_minus[i] -= eps
                grad_fd = (game.payoff_functions[i](*u_plus) - game.payoff_functions[i](*u_minus)) / (2 * eps)
                self.assertAlmostEqual(grad_fd, 0.0, places=4)

    def test_diagonal_symmetrization_m_greater_than_1(self):
        N = 2
        m = 2
        games = generate_quadratic_games(ensemble_size=3, N=N, m=m, seed=55)
        for game in games:
            self.assertEqual(game.H.shape, (N * m, N * m))
            for i in range(N):
                s, e = i * m, (i + 1) * m
                H_ii = game.H[s:e, s:e]
                np.testing.assert_allclose(H_ii, H_ii.T, atol=1e-12)

            zeros = [0.0] * (N * m)
            for f in game.payoff_functions:
                val = f(*zeros)
                self.assertIsInstance(val, float)

    def test_reproducibility(self):
        g1 = generate_quadratic_games(ensemble_size=3, N=2, m=1, seed=777)
        g2 = generate_quadratic_games(ensemble_size=3, N=2, m=1, seed=777)
        g3 = generate_quadratic_games(ensemble_size=3, N=2, m=1, seed=888)

        for i in range(3):
            np.testing.assert_array_equal(g1[i].H, g2[i].H)
            np.testing.assert_array_equal(g1[i].d, g2[i].d)
            np.testing.assert_array_equal(g1[i].c, g2[i].c)
            np.testing.assert_array_equal(g1[i].nash_equilibrium, g2[i].nash_equilibrium)

        self.assertFalse(np.allclose(g1[0].H, g3[0].H))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            QuadraticGameGenerator(N=1)
        with self.assertRaises(ValueError):
            QuadraticGameGenerator(N=2, m=0)
        with self.assertRaises(ValueError):
            QuadraticGameGenerator(N=2, alpha=0.0)
        with self.assertRaises(ValueError):
            QuadraticGameGenerator(N=2, action_lower_interval=(0.1, 0.5))
        with self.assertRaises(ValueError):
            QuadraticGameGenerator(N=2, action_upper_interval=(-0.5, 0.5))

    def test_convergence_zo_gradient_play(self):
        games = generate_quadratic_games(ensemble_size=1, N=2, m=1, alpha=0.05, seed=10)
        game = games[0]

        np.random.seed(42)
        estimator = ZOGrandientPlay(game, v=5.0, a=0.8, epsilon=1e-4)
        initial_actions = np.array([[0.0], [0.0]])
        history = estimator.estimate(initial_actions, num_steps=6000)

        est_flat = np.mean(history.state_history[-50:], axis=0).ravel()
        true_flat = game.nash_equilibrium.ravel()
        error = np.linalg.norm(est_flat - true_flat)
        self.assertLess(error, 0.15)

    def test_convergence_ne_seeking(self):
        games = generate_quadratic_games(ensemble_size=1, N=2, m=1, alpha=0.05, seed=10)
        game = games[0]

        estimator = NESeeking(
            game=game,
            k=np.array([[5.0], [5.0]]),
            a=np.array([[0.15], [0.15]]),
            omega=np.array([[25.0], [31.0]]),
            dt=0.005,
        )
        initial_u_hat = np.array([[0.0], [0.0]])
        history = estimator.estimate(initial_u_hat, num_steps=20000)

        est_flat = history.state_history[-1].ravel()
        true_flat = game.nash_equilibrium
        error = np.linalg.norm(est_flat - true_flat)
        self.assertLess(error, 0.15)


if __name__ == "__main__":
    unittest.main()
