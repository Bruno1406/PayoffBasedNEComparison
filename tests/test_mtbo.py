import csv
import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from payoffnash import (
    MetaParamOptimization,
    NESeeking,
    ZOGrandientPlay,
    generate_quadratic_games,
)


class TestMTBO(unittest.TestCase):
    def setUp(self):
        self.games = generate_quadratic_games(ensemble_size=2, N=2, m=1, alpha=0.05, seed=42)

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
        self.assertEqual(ne_specs["phi"], (2, 1))

    def test_broadcasting_and_flattening(self):
        game = self.games[0]
        search_space = {"k": (1.0, 5.0), "a": (0.05, 0.2)}
        fixed_params = {
            "omega": np.array([[20.0], [30.0]]),
            "phi": np.array([[0.0], [0.0]]),
            "dt": 0.01,
        }

        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=NESeeking,
            fixed_params=fixed_params,
            n_init=2,
            n_bo_steps=1,
            num_iters=10,
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

    def test_mtbo_optimization_zo_gradient_play(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "zo_results.csv")
            search_space = {"v": (2.0, 5.0), "a": (0.5, 2.0)}
            fixed_params = {"estimation_type": "two-point", "epsilon": 1e-4}

            optimizer = MetaParamOptimization(
                game_ensemble=self.games,
                search_space=search_space,
                algorithm_class=ZOGrandientPlay,
                output_path=csv_path,
                num_iters=30,
                smoothing_window=10,
                n_init=3,
                n_bo_steps=2,
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
                "phi": np.array([[0.0], [0.0]]),
                "dt": 0.005,
            }

            optimizer = MetaParamOptimization(
                game_ensemble=self.games,
                search_space=search_space,
                algorithm_class=NESeeking,
                output_path=csv_path,
                num_iters=40,
                smoothing_window=10,
                n_init=3,
                n_bo_steps=2,
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

    def test_single_task_evaluation(self):
        search_space = {"k": (1.0, 5.0), "a": (0.05, 0.2)}
        fixed_params = {
            "omega": np.array([[20.0], [30.0]]),
            "phi": np.array([[0.0], [0.0]]),
            "dt": 0.01,
        }
        optimizer = MetaParamOptimization(
            game_ensemble=self.games,
            search_space=search_space,
            algorithm_class=NESeeking,
            num_iters=20,
            smoothing_window=5,
            fixed_params=fixed_params,
        )
        theta = np.array([2.5, 0.1])
        dist_0 = optimizer.evaluate_task(theta, 0)
        dist_1 = optimizer.evaluate_task(theta, 1)
        self.assertIsInstance(dist_0, float)
        self.assertIsInstance(dist_1, float)
        self.assertGreater(dist_0, 0.0)
        self.assertGreater(dist_1, 0.0)

        dists = optimizer.evaluate_ensemble(theta)
        self.assertEqual(len(dists), 2)
        np.testing.assert_allclose(dists[0], dist_0)
        np.testing.assert_allclose(dists[1], dist_1)



if __name__ == "__main__":
    unittest.main()

