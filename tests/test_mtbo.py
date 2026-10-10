import csv
import os
import sys
import tempfile
import unittest
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from payoffnash import (
    EstimateHistory,
    MetaParamOptimization,
    NESeeking,
    QuadraticGame,
    ZOGrandientPlay,
    generate_quadratic_games,
)


class TestMTBO(unittest.TestCase):
    def setUp(self):
        self.games = generate_quadratic_games(ensemble_size=2, N=2, m=1, alpha=0.05, seed=42)

    @staticmethod
    def _dummy_cost(game: QuadraticGame, history: EstimateHistory) -> float:
        x_star = game.nash_equilibrium.ravel()
        dists = np.linalg.norm(history.action_history - x_star, axis=-1)
        k_window = min(10, len(dists))
        return float(np.mean(dists[-k_window:]))

    def test_param_specs_introspection(self):
        game = self.games[0]
        zo_specs = ZOGrandientPlay.get_param_specs(game)
        self.assertEqual(zo_specs["v"], (1,))
        self.assertEqual(zo_specs["a"], (1,))
        self.assertEqual(zo_specs["epsilon"], (1,))

        ne_specs = NESeeking.get_param_specs(game)
        self.assertEqual(ne_specs["k"], (2, 1))
        self.assertEqual(ne_specs["a"], (2, 1))
        self.assertEqual(ne_specs["omega"], (2, 1))
        self.assertEqual(ne_specs["omega_h"], (2,))

    def test_broadcasting_and_flattening(self):
        game = self.games[0]
        search_space = {"k": (1.0, 5.0), "a": (0.05, 0.2)}
        fixed_params = {
            "omega": np.array([[20.0], [30.0]]),
            "dt": 0.01,
        }

        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=NESeeking,
            cost_function=self._dummy_cost,
            fixed_params=fixed_params,
            n_init=2,
            max_bo_steps=1,
            num_steps=10,
        )

        theta = np.array([2.5, 0.1])
        kwargs = optimizer._build_estimator_kwargs(theta, game)

        self.assertEqual(kwargs["k"].shape, (2, 1))
        np.testing.assert_allclose(kwargs["k"], np.array([[2.5], [2.5]]))
        self.assertEqual(kwargs["a"].shape, (2, 1))
        np.testing.assert_allclose(kwargs["a"], np.array([[0.1], [0.1]]))

        flat = optimizer._flatten_hyperparams(theta, game)
        self.assertIn("k[0,0]", flat)
        self.assertIn("k[1,0]", flat)
        self.assertIn("omega[0,0]", flat)
        self.assertIn("omega[1,0]", flat)
        self.assertEqual(flat["k[0,0]"], 2.5)
        self.assertEqual(flat["omega[1,0]"], 30.0)

    def test_indexed_parameter_parsing(self):
        game = self.games[0]
        search_space = {
            "omega[0,0]": (20.0, 30.0),
            "omega[1,0]": (31.0, 40.0),
            "k": (1.0, 5.0),
        }
        fixed_params = {
            "a": np.array([[0.1], [0.1]]),
            "dt": 0.01,
        }

        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=NESeeking,
            cost_function=self._dummy_cost,
            fixed_params=fixed_params,
            n_init=2,
            max_bo_steps=1,
            num_steps=10,
        )

        theta = np.array([25.0, 35.0, 2.0])
        kwargs = optimizer._build_estimator_kwargs(theta, game)

        self.assertEqual(kwargs["omega"].shape, (2, 1))
        self.assertEqual(kwargs["omega"][0, 0], 25.0)
        self.assertEqual(kwargs["omega"][1, 0], 35.0)
        np.testing.assert_allclose(kwargs["k"], np.array([[2.0], [2.0]]))

    def test_single_task_evaluation(self):
        search_space = {"k": (1.0, 5.0), "a": (0.05, 0.2)}
        fixed_params = {
            "omega": np.array([[20.0], [30.0]]),
            "dt": 0.01,
        }
        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=NESeeking,
            cost_function=self._dummy_cost,
            num_steps=20,
            fixed_params=fixed_params,
        )
        theta = np.array([2.5, 0.1])
        dist_0 = optimizer.evaluate_task(theta, 0)
        dist_1 = optimizer.evaluate_task(theta, 1)
        self.assertIsInstance(dist_0, float)
        self.assertIsInstance(dist_1, float)
        self.assertGreater(dist_0, 0.0)
        self.assertGreater(dist_1, 0.0)

    def test_single_task_covariance_equivalence(self):
        single_game = [self.games[0]]
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        fixed_params = {"epsilon": 1e-4}
        optimizer = MetaParamOptimization(
            game_ensemble=single_game,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=4,
            max_bo_steps=2,
            min_bo_steps=1,
            seed=42,
            fixed_params=fixed_params,
        )
        optimizer.run_optimization_loop()

        train_x = optimizer.train_x
        train_y = optimizer.train_y
        model = MultiTaskGP(train_x, train_y, task_feature=-1, outcome_transform=Standardize(m=1))
        test_cands = torch.rand(10, 2, dtype=torch.double)

        cands_batched = torch.cat([
            test_cands.unsqueeze(1),
            torch.zeros(10, 1, 1, dtype=torch.double)
        ], dim=-1)
        post_batched = model.posterior(cands_batched)
        var_avg = post_batched.mvn.covariance_matrix.sum(dim=(-1, -2)) / (1.0 ** 2)

        cands_direct = torch.cat([test_cands, torch.zeros(10, 1, dtype=torch.double)], dim=-1)
        post_direct = model.posterior(cands_direct)
        var_direct = post_direct.variance.squeeze(-1)

        torch.testing.assert_close(var_avg, var_direct, atol=1e-6, rtol=1e-6)

    def test_direct_task_selection_equality(self):
        num_tasks = 2
        train_x = torch.tensor([
            [0.2, 0.3, 0.0],
            [0.7, 0.8, 1.0],
            [0.4, 0.5, 0.0],
            [0.9, 0.1, 1.0],
        ], dtype=torch.double)
        train_y = torch.tensor([[1.0], [2.0], [1.5], [2.5]], dtype=torch.double)
        model = MultiTaskGP(train_x, train_y, task_feature=-1, outcome_transform=Standardize(m=1))

        cands = torch.tensor([[0.5, 0.5], [0.1, 0.9]], dtype=torch.double)
        cands_batched = torch.cat([
            cands.unsqueeze(1).expand(2, 2, 2),
            torch.arange(2, dtype=torch.double).view(1, 2, 1).expand(2, 2, 1),
        ], dim=-1)
        post = model.posterior(cands_batched)
        means = post.mean.squeeze(-1)
        variances = torch.diagonal(post.mvn.covariance_matrix, dim1=-2, dim2=-1)

        best_cand_idx = 0
        star_means = means[best_cand_idx]
        star_stds = torch.sqrt(torch.clamp(variances[best_cand_idx], min=1e-12))

        cand_slice = torch.cat([
            cands[best_cand_idx].repeat(num_tasks, 1),
            torch.arange(num_tasks, dtype=torch.double).unsqueeze(-1)
        ], dim=-1)
        post_slice = model.posterior(cand_slice)
        separate_means = post_slice.mean.squeeze(-1)
        separate_stds = torch.sqrt(torch.clamp(post_slice.variance.squeeze(-1), min=1e-12))

        torch.testing.assert_close(star_means, separate_means, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(star_stds, separate_stds, atol=1e-12, rtol=1e-12)

    def test_zero_duplicate_queries(self):
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=2,
            max_bo_steps=4,
            eps_ei=None,
            seed=42,
            fixed_params={"epsilon": 1e-4},
        )
        optimizer.run_optimization_loop()
        unique_cands = torch.unique(optimizer.train_x[:, :-1], dim=0)
        self.assertEqual(len(unique_cands), len(optimizer.train_x))

    def test_dimension_scaled_ceiling_heuristic(self):
        search_2d = {"p1": (0.0, 1.0), "p2": (0.0, 1.0)}
        search_6d = {f"p{i}": (0.0, 1.0) for i in range(6)}

        # M=1, d=2 -> max(30, 25*2) = 50
        opt_1_2 = MetaParamOptimization(
            [self.games[0]], search_2d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_1_2.max_bo_steps, 50)
        self.assertEqual(opt_1_2.min_bo_steps, 10)

        # M=1, d=6 -> max(30, 25*6) = 150
        opt_1_6 = MetaParamOptimization(
            [self.games[0]], search_6d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_1_6.max_bo_steps, 150)
        self.assertEqual(opt_1_6.min_bo_steps, 10)

        # M=100 games
        games_100 = [self.games[0]] * 100
        # M=100, d=2 -> min(100 * max(5, 4), 1500) = 500
        opt_100_2 = MetaParamOptimization(
            games_100, search_2d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_100_2.max_bo_steps, 500)
        self.assertEqual(opt_100_2.min_bo_steps, 50)

        # M=100, d=6 -> min(100 * max(5, 12), 1500) = 1200
        opt_100_6 = MetaParamOptimization(
            games_100, search_6d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_100_6.max_bo_steps, 1200)
        self.assertEqual(opt_100_6.min_bo_steps, 50)

    def test_adaptive_garbage_collection_frequency(self):
        search_2d = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        opt_single = MetaParamOptimization(
            [self.games[0]], search_2d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_single.gc_interval, 100)

        games_100 = [self.games[0]] * 100
        opt_100 = MetaParamOptimization(
            games_100, search_2d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_100.gc_interval, 25)

    def test_mathematical_ei_convergence(self):
        single_game = [self.games[0]]
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}

        optimizer = MetaParamOptimization(
            game_ensemble=single_game,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=15,
            n_init=5,
            min_bo_steps=2,
            eps_ei=1e-4,
            ei_window=3,
            max_bo_steps=30,
            fixed_params={"epsilon": 1e-4},
            seed=42,
        )
        optimizer.run_optimization_loop()
        self.assertTrue(optimizer.stopped_early)
        self.assertEqual(optimizer.stop_reason, "ei_convergence")
        self.assertLess(optimizer.n_bo_steps_evaluated, 30)

    def test_hard_safety_ceiling_max_bo_steps(self):
        single_game = [self.games[0]]
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}

        def flat_cost(g, h):
            return 10.0

        optimizer = MetaParamOptimization(
            game_ensemble=single_game,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=flat_cost,
            num_steps=10,
            n_init=2,
            min_bo_steps=2,
            eps_ei=1e-15,
            max_bo_steps=4,
            fixed_params={"epsilon": 1e-4},
            seed=42,
        )
        optimizer.run_optimization_loop()
        self.assertFalse(optimizer.stopped_early)
        self.assertEqual(optimizer.stop_reason, "max_bo_steps_reached")
        self.assertEqual(optimizer.n_bo_steps_evaluated, 4)

    def test_persistent_model_warm_starting(self):
        train_x = torch.tensor([[0.2, 0.3, 0.0], [0.7, 0.8, 1.0]], dtype=torch.double)
        train_y = torch.tensor([[1.0], [2.0]], dtype=torch.double)
        model = MultiTaskGP(train_x, train_y, task_feature=-1, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 10}})

        new_x = torch.cat([train_x, torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.double)], dim=0)
        new_y = torch.cat([train_y, torch.tensor([[1.5]], dtype=torch.double)], dim=0)

        # Warm start
        model.outcome_transform.train()
        transformed_y, _ = model.outcome_transform(new_y, X=new_x)
        model.outcome_transform.eval()
        model.set_train_data(new_x, transformed_y.squeeze(-1), strict=False)
        model.train()
        model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 10}})
        model.eval()
        self.assertEqual(model.train_inputs[0].shape[0], 3)

    def test_oneshot_post_bo_verification_safeguard(self):
        search_space = {"k": (1.0, 5.0), "a": (0.05, 0.2)}
        fixed_params = {
            "omega": np.array([[20.0], [30.0]]),
            "dt": 0.01,
        }
        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=NESeeking,
            cost_function=self._dummy_cost,
            num_steps=20,
            n_init=2,
            max_bo_steps=2,
            min_bo_steps=1,
            seed=42,
            fixed_params=fixed_params,
        )
        results = optimizer.run_optimization_loop()
        for j in range(len(self.games)):
            res_val = results[j]["objective_value"]
            self.assertLessEqual(res_val, optimizer.task_anchors[j] + 1e-9)
            eval_val = optimizer.evaluate_task(optimizer.best_thetas[j], j)
            self.assertAlmostEqual(eval_val, res_val, places=5)

    def test_validation_errors(self):
        single_game = [self.games[0]]
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        with self.assertRaises(ValueError):
            MetaParamOptimization([], search_space, ZOGrandientPlay, self._dummy_cost)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, {}, ZOGrandientPlay, self._dummy_cost)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, num_steps=0)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, n_init=0)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, max_bo_steps=-1)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, min_bo_steps=-1)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, eps_ei=-0.01)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, ei_window=0)

    def test_mtbo_optimization_zo_gradient_play(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "zo_results.csv")
            search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
            fixed_params = {"epsilon": 1e-4}

            optimizer = MetaParamOptimization(
                game_ensemble=self.games,
                search_space=search_space,
                algorithm_class=ZOGrandientPlay,
                cost_function=self._dummy_cost,
                output_path=csv_path,
                num_steps=30,
                n_init=3,
                max_bo_steps=2,
                min_bo_steps=1,
                seed=123,
                fixed_params=fixed_params,
            )

            results = optimizer.run_optimization_loop()
            self.assertEqual(len(results), len(self.games) + 1)

            for i in range(len(self.games)):
                res = results[i]
                self.assertEqual(res["game_id"], f"game_{i}")
                self.assertFalse(res["is_default"])
                self.assertGreaterEqual(res["v"], 2.0)
                self.assertLessEqual(res["v"], 5.0)
                self.assertGreaterEqual(res["a"], 0.5)
                self.assertLessEqual(res["a"], 2.0)
                self.assertIn("objective_value", res)
                self.assertGreater(res["objective_value"], 0.0)

            default_res = results[-1]
            self.assertEqual(default_res["game_id"], "default")
            self.assertTrue(default_res["is_default"])
            self.assertGreaterEqual(default_res["v"], 2.0)
            self.assertLessEqual(default_res["v"], 5.0)

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0]["game_id"], "game_0")
                self.assertEqual(rows[-1]["game_id"], "default")

    def test_mtbo_optimization_ne_seeking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ne_results.csv")
            search_space = {"k": (1.0, 4.0), "a": (0.05, 0.15)}
            fixed_params = {
                "omega": np.array([[25.0], [31.0]]),
                "dt": 0.005,
            }

            optimizer = MetaParamOptimization(
                game_ensemble=self.games,
                search_space=search_space,
                algorithm_class=NESeeking,
                cost_function=self._dummy_cost,
                output_path=csv_path,
                num_steps=40,
                n_init=3,
                max_bo_steps=2,
                min_bo_steps=1,
                seed=456,
                fixed_params=fixed_params,
            )

            results = optimizer.run_optimization_loop()
            self.assertEqual(len(results), 3)

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                self.assertIn("k[0,0]", headers)
                self.assertIn("k[1,0]", headers)
                self.assertIn("omega[0,0]", headers)
                self.assertIn("objective_value", headers)
                rows = list(reader)
                self.assertEqual(len(rows), 3)

    def test_multitask_incumbent_average_not_single_task_min(self):
        def heterogeneous_cost(game: QuadraticGame, history: EstimateHistory) -> float:
            base = float(np.mean(np.linalg.norm(history.state_history - game.nash_equilibrium, axis=-1)))
            offset = 10.0 if np.allclose(game.nash_equilibrium, self.games[1].nash_equilibrium) else 0.5
            return base + offset

        search_space = {"v": (0.5, 5.0), "a": (2.0, 10.0)}
        fixed_params = {"epsilon": 1e-4}

        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=heterogeneous_cost,
            num_steps=30,
            n_init=4,
            max_bo_steps=2,
            min_bo_steps=1,
            seed=42,
            fixed_params=fixed_params,
        )

        results = optimizer.run_optimization_loop()
        min_single_task = float(optimizer.train_y.min().item())
        self.assertGreater(results[-1]["objective_value"], min_single_task + 1.0)


if __name__ == "__main__":
    unittest.main()
