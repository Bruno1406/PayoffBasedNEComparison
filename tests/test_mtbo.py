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

    def test_single_task_block_design(self):
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
            seed=42,
            fixed_params=fixed_params,
        )
        results = optimizer.run_optimization_loop()
        self.assertEqual(optimizer.train_x.shape[1], optimizer.dim + 1)
        self.assertEqual(optimizer.train_y.shape[1], 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["game_id"], "game_0")
        self.assertEqual(results[1]["game_id"], "default")

    def test_multitask_design_shapes(self):
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=3,
            max_bo_steps=2,
            seed=42,
            fixed_params={"epsilon": 1e-4},
        )
        results = optimizer.run_optimization_loop()
        self.assertEqual(optimizer.train_x.shape[1], optimizer.dim + 1)
        self.assertEqual(optimizer.train_y.shape[1], 1)
        self.assertEqual(len(optimizer.train_x), len(optimizer.train_y))
        self.assertEqual(len(results), optimizer.n_tasks + 1)

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
        unique_cands = torch.unique(optimizer.train_x, dim=0)
        self.assertEqual(len(unique_cands), len(optimizer.train_x))

    def test_dimension_scaled_ceiling_heuristic(self):
        search_2d = {"p1": (0.0, 1.0), "p2": (0.0, 1.0)}
        search_6d = {f"p{i}": (0.0, 1.0) for i in range(6)}

        opt_2d = MetaParamOptimization(
            [self.games[0]], search_2d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_2d.max_bo_steps, 200)
        self.assertEqual(opt_2d.n_init, 10)

        opt_6d = MetaParamOptimization(
            [self.games[0]], search_6d, ZOGrandientPlay, self._dummy_cost, num_steps=10
        )
        self.assertEqual(opt_6d.max_bo_steps, 200)
        self.assertEqual(opt_6d.n_init, 12)

    def test_non_negative_tunability_guarantee(self):
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=15,
            n_init=4,
            max_bo_steps=3,
            seed=42,
            fixed_params={"epsilon": 1e-4},
        )
        results = optimizer.run_optimization_loop()
        for j in range(len(self.games)):
            tailored_val = results[j]["objective_value"]
            y_def_j = optimizer.evaluate_task(optimizer.best_theta_def, j)
            self.assertLessEqual(tailored_val, y_def_j + 1e-9)
            tunability_gain = y_def_j - tailored_val
            self.assertGreaterEqual(tunability_gain, -1e-9)

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
            eps_ei=1e-4,
            max_bo_steps=30,
            fixed_params={"epsilon": 1e-4},
            seed=42,
        )
        optimizer.run_optimization_loop()
        self.assertTrue(optimizer.stopped_early)
        self.assertEqual(optimizer.stop_reason, "all_tasks_pruned")
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
            eps_ei=1e-15,
            max_bo_steps=4,
            fixed_params={"epsilon": 1e-4},
            seed=42,
        )
        optimizer.run_optimization_loop()
        self.assertFalse(optimizer.stopped_early)
        self.assertEqual(optimizer.stop_reason, "max_bo_steps_reached")
        self.assertEqual(optimizer.n_bo_steps_evaluated, 4)

    def test_evaluate_ensemble(self):
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
            num_steps=10,
            fixed_params=fixed_params,
        )
        theta = np.array([2.5, 0.1])
        losses = optimizer.evaluate_ensemble(theta)
        self.assertIsInstance(losses, np.ndarray)
        self.assertEqual(losses.shape, (len(self.games),))
        self.assertAlmostEqual(losses[0], optimizer.evaluate_task(theta, 0))
        self.assertAlmostEqual(losses[1], optimizer.evaluate_task(theta, 1))

    def test_exact_empirical_ground_truth(self):
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
            seed=42,
            fixed_params=fixed_params,
        )
        results = optimizer.run_optimization_loop()
        train_y_np = optimizer.train_y.cpu().numpy()
        train_x_np = optimizer.train_x.cpu().numpy()
        for j in range(len(self.games)):
            res_val = results[j]["objective_value"]
            task_j_mask = (train_x_np[:, -1] == j)
            min_seen = float(np.min(train_y_np[task_j_mask]))
            self.assertLessEqual(res_val, min_seen + 1e-9)
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
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, patience=0)
        with self.assertRaises(ValueError):
            MetaParamOptimization(single_game, search_space, ZOGrandientPlay, self._dummy_cost, eps_ei=-0.01)

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
            seed=42,
            fixed_params=fixed_params,
        )

        results = optimizer.run_optimization_loop()
        min_single_task = float(optimizer.train_y.min().item())
        self.assertGreater(results[-1]["objective_value"], min_single_task + 1.0)

    def test_log_scale_mapping(self):
        search_space = {"p1": (1e-3, 10.0), "p2": (1.0, 100.0)}
        opt = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
        )
        self.assertTrue(opt.is_log["p1"])
        self.assertTrue(opt.is_log["p2"])

        lower_orig = np.array([1e-3, 1.0])
        upper_orig = np.array([10.0, 100.0])
        np.testing.assert_allclose(opt._normalize(lower_orig), np.array([0.0, 0.0]), atol=1e-7)
        np.testing.assert_allclose(opt._normalize(upper_orig), np.array([1.0, 1.0]), atol=1e-7)

        mid_unit = np.array([0.5, 0.5])
        mid_orig = opt._unnormalize(mid_unit)
        np.testing.assert_allclose(opt._normalize(mid_orig), mid_unit, atol=1e-7)

    def test_surrogate_validation_metrics(self):
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=3,
            max_bo_steps=1,
            seed=42,
            fixed_params={"epsilon": 1e-4},
        )
        with self.assertRaises(RuntimeError):
            optimizer.validate()

        optimizer.run_optimization_loop()
        metrics = optimizer.validate()

        self.assertIn("loo_rmse", metrics)
        self.assertIn("loo_smse", metrics)
        self.assertIn("calibration_std", metrics)
        self.assertIn("suboptimality_bounds", metrics)
        self.assertIn("anomalies_detected", metrics)
        self.assertIn("refinements_executed", metrics)

        self.assertTrue(np.isfinite(metrics["loo_rmse"]))
        self.assertTrue(np.isfinite(metrics["loo_smse"]))
        self.assertTrue(np.isfinite(metrics["calibration_std"]))
        self.assertGreaterEqual(metrics["loo_rmse"], 0.0)
        self.assertGreaterEqual(metrics["calibration_std"], 0.0)
        self.assertEqual(len(metrics["suboptimality_bounds"]), len(self.games))
        self.assertIs(optimizer.validation_metrics, metrics)

    def test_parallel_multi_core_simulation(self):
        games = generate_quadratic_games(ensemble_size=3, N=2, m=1, alpha=0.05, seed=42)
        search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
        opt = MetaParamOptimization(
            game_ensemble=games,
            search_space=search_space,
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=2,
            max_bo_steps=2,
            n_jobs=2,
            seed=42,
            fixed_params={"epsilon": 1e-4},
        )
        results = opt.run_optimization_loop()
        self.assertEqual(len(results), 4)
        self.assertEqual(opt.n_jobs, 2)

    def test_progress_logging_and_pruning_reasons(self):
        import io
        from contextlib import redirect_stdout

        # Test _compute_stagnation_and_patience
        stagnant, rem_pat = MetaParamOptimization._compute_stagnation_and_patience([1.0], patience=5, tol=1e-4)
        self.assertEqual(stagnant, 0)
        self.assertEqual(rem_pat, 5)

        stagnant, rem_pat = MetaParamOptimization._compute_stagnation_and_patience([1.0, 1.0, 1.0], patience=5, tol=1e-4)
        self.assertEqual(stagnant, 2)
        self.assertEqual(rem_pat, 3)

        games = generate_quadratic_games(ensemble_size=2, N=2, m=1, alpha=0.05, seed=42)
        buf = io.StringIO()
        with redirect_stdout(buf):
            opt = MetaParamOptimization(
                game_ensemble=games,
                search_space={"v": (1.0, 5.0), "a": (0.5, 2.0)},
                algorithm_class=ZOGrandientPlay,
                cost_function=self._dummy_cost,
                num_steps=10,
                n_init=2,
                max_bo_steps=5,
                min_bo_steps=1,
                eps_ei=100.0,  # Ensure pruning triggers immediately
                patience=2,
                tol_stagnation=1e-4,
                n_jobs=1,
                seed=42,
                fixed_params={"epsilon": 1e-4},
            )
            opt.run_optimization_loop()

        output = buf.getvalue()
        self.assertIn("[BO Step 1/", output)
        self.assertIn("[Task Pruned]", output)
        self.assertIn("Expected Improvement", output)
        self.assertIn("EI =", output)
        self.assertTrue(opt.stopped_early)
        self.assertEqual(opt.stop_reason, "all_tasks_pruned")

        # Test 5-step progress table
        buf5 = io.StringIO()
        with redirect_stdout(buf5):
            opt5 = MetaParamOptimization(
                game_ensemble=games,
                search_space={"v": (1.0, 5.0), "a": (0.5, 2.0)},
                algorithm_class=ZOGrandientPlay,
                cost_function=self._dummy_cost,
                num_steps=10,
                n_init=2,
                max_bo_steps=5,
                eps_ei=None,
                patience=10,
                n_jobs=1,
                seed=42,
                fixed_params={"epsilon": 1e-4},
            )
            opt5.run_optimization_loop()

        out5 = buf5.getvalue()
        self.assertIn("BO Step 5/5 Progress Summary", out5)
        self.assertIn("Universal Default", out5)
        self.assertIn("Game 0", out5)
        self.assertIn("Game 1", out5)

    def test_scaled_params_kwargs_construction(self):
        import inspect
        if "scaled_params" not in inspect.signature(MetaParamOptimization.__init__).parameters:
            self.skipTest("scaled_params not yet implemented in MetaParamOptimization")

        game = self.games[0]
        w0 = float(game.action_sets[0][0][1] - game.action_sets[0][0][0])
        w1 = float(game.action_sets[1][0][1] - game.action_sets[1][0][0])
        w_min = min(w0, w1)

        # 1. Test ZOGrandientPlay scalar scaling by min_box_width
        zo_opt = MetaParamOptimization(
            game_ensemble=[game],
            search_space={"v": (0.01, 100.0), "a": (0.005, 2.0)},
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_jobs=1,
            scaled_params={"a": "min_box_width"},
            fixed_params={"epsilon": 1e-4},
        )
        theta_zo = np.array([5.0, 0.4])
        kwargs_zo = zo_opt._build_estimator_kwargs(theta_zo, game)
        self.assertAlmostEqual(kwargs_zo["v"], 5.0)
        self.assertAlmostEqual(kwargs_zo["a"], 0.4 * w_min)

        # 2. Test NESeeking player-specific scaling by player_box_width
        ne_opt = MetaParamOptimization(
            game_ensemble=[game],
            search_space={
                "k[0,0]": (0.01, 100.0),
                "k[1,0]": (0.01, 100.0),
                "a[0,0]": (0.005, 2.0),
                "a[1,0]": (0.005, 2.0),
                "omega[0,0]": (1.0, 50.0),
                "omega[1,0]": (1.0, 50.0),
            },
            algorithm_class=NESeeking,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_jobs=1,
            scaled_params={"a[0,0]": "player_box_width", "a[1,0]": "player_box_width"},
            fixed_params={"dt": 0.001, "omega_h": np.array([10.0, 10.0])},
        )
        theta_ne = np.array([10.0, 20.0, 0.3, 0.5, 15.0, 25.0])
        kwargs_ne = ne_opt._build_estimator_kwargs(theta_ne, game)
        self.assertAlmostEqual(float(kwargs_ne["k"][0, 0]), 10.0)
        self.assertAlmostEqual(float(kwargs_ne["k"][1, 0]), 20.0)
        self.assertAlmostEqual(float(kwargs_ne["a"][0, 0]), 0.3 * w0)
        self.assertAlmostEqual(float(kwargs_ne["a"][1, 0]), 0.5 * w1)

    def test_scaled_params_optimization_loop(self):
        games = generate_quadratic_games(ensemble_size=2, N=2, m=1, alpha=0.05, seed=42)
        opt = MetaParamOptimization(
            game_ensemble=games,
            search_space={"v": (0.01, 100.0), "a": (0.005, 2.0)},
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=2,
            max_bo_steps=2,
            eps_ei=None,
            patience=5,
            n_jobs=1,
            seed=42,
            scaled_params={"a": "min_box_width"},
            fixed_params={"epsilon": 1e-4},
        )
        results = opt.run_optimization_loop()
        self.assertEqual(len(results), len(games) + 1)
        self.assertEqual(len(opt.best_thetas), len(games))
        self.assertIsNotNone(opt.best_theta_def)
        # Verify that evaluating task works with scaled_params
        val0 = opt.evaluate_task(opt.best_thetas[0], 0)
        self.assertTrue(np.isfinite(val0))

    def test_sim_per_param_averaging(self):
        games = generate_quadratic_games(ensemble_size=1, N=2, m=1, alpha=0.05, seed=42)
        opt_single = MetaParamOptimization(
            game_ensemble=games,
            search_space={"v": (0.01, 100.0), "a": (0.005, 2.0)},
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_jobs=1,
            seed=42,
            sim_per_param=1,
            fixed_params={"epsilon": 1e-4},
        )
        opt_multi = MetaParamOptimization(
            game_ensemble=games,
            search_space={"v": (0.01, 100.0), "a": (0.005, 2.0)},
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_jobs=1,
            seed=42,
            sim_per_param=3,
            fixed_params={"epsilon": 1e-4},
        )
        theta = np.array([5.0, 0.5])
        val1 = opt_single.evaluate_task(theta, 0)
        val3 = opt_multi.evaluate_task(theta, 0)
        self.assertTrue(np.isfinite(val1))
        self.assertTrue(np.isfinite(val3))

    def test_min_bo_steps_guards_pruning(self):
        games = generate_quadratic_games(ensemble_size=2, N=2, m=1, alpha=0.05, seed=42)
        opt = MetaParamOptimization(
            game_ensemble=games,
            search_space={"v": (0.01, 100.0), "a": (0.005, 2.0)},
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=2,
            max_bo_steps=5,
            min_bo_steps=4,
            eps_ei=1e6,  # Very high EI threshold that would prune immediately without min_bo_steps
            patience=1,
            n_jobs=1,
            seed=42,
            fixed_params={"epsilon": 1e-4},
        )
        results = opt.run_optimization_loop()
        self.assertGreaterEqual(opt.n_bo_steps_evaluated, 4)
        self.assertEqual(len(results), len(games) + 1)

    def test_run_parallel_single_task_bo(self):
        games = generate_quadratic_games(ensemble_size=2, N=2, m=1, alpha=0.05, seed=42)
        opt = MetaParamOptimization(
            game_ensemble=games,
            search_space={"v": (0.01, 100.0), "a": (0.005, 2.0)},
            algorithm_class=ZOGrandientPlay,
            cost_function=self._dummy_cost,
            num_steps=10,
            n_init=2,
            max_bo_steps=2,
            min_bo_steps=1,
            eps_ei=None,
            patience=5,
            n_jobs=1,
            seed=42,
            fixed_params={"epsilon": 1e-4},
        )
        results = opt.run_parallel_single_task_bo(sim_per_param=2)
        self.assertEqual(opt.sim_per_param, 2)
        self.assertEqual(len(results), len(games))
        self.assertEqual(len(opt.best_thetas), len(games))
        for res in results:
            self.assertFalse(res["is_default"])
            self.assertTrue(np.isfinite(res["objective_value"]))


if __name__ == "__main__":
    unittest.main()

