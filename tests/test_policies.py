import http
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler

import numpy as np

from hopes.fun_utils import piecewise_linear
from hopes.policy.policies import (
    ClassificationBasedPolicy,
    HttpPolicy,
    PiecewiseLinearPolicy,
    RandomPolicy,
)
from tests.action_probs_utils import generate_action_probs


class TestPolicies(unittest.TestCase):
    def test_rnd_policy(self):
        rnd_policy = RandomPolicy(num_actions=3)
        log_probs = rnd_policy.log_likelihoods(obs=np.random.rand(10, 5))
        self.assert_log_probs(log_probs, expected_shape=(10, 3))

    def test_logistic_based_policy(self):
        self._test_classification_policy("logistic")

    def test_mlp_based_policy(self):
        self._test_classification_policy("mlp")

    def _test_classification_policy(self, model_type: str) -> None:
        # generate a random dataset of (obs, act) for target policy
        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)

        # create and fit a classification-based policy
        reg_policy = ClassificationBasedPolicy(obs=obs, act=act, classification_model=model_type)
        reg_policy.fit()

        # check if the policy returns valid log-likelihoods
        new_obs = np.random.rand(10, num_obs)
        act_probs = reg_policy.compute_action_probs(obs=new_obs)
        self.assert_act_probs(act_probs, expected_shape=(10, num_actions))

        actions = reg_policy.select_action(obs=new_obs)
        self.assertIsInstance(actions, np.ndarray)
        self.assertEqual(len(actions), 10)
        self.assertTrue(all(0 <= action < num_actions for action in actions))

    def test_piecewise_linear_policy(self):
        # define a simple piecewise linear function that simulates a classic outdoor air reset control
        # outdoor temperatures
        obs = np.arange(-10, 30, 0.1)
        # supply air temperatures
        act = piecewise_linear(obs, y0=30, y1=15, left_cp=10, right_cp=20, slope=-0.5)

        # create and fit a piecewise linear policy
        reg_policy = PiecewiseLinearPolicy(obs=obs, act=act, actions_bins=list(range(15, 31)))
        reg_policy.fit()

        # check if the policy returns valid log-likelihoods
        new_obs = np.random.randint(-10, 30, 10).reshape(-1, 1)
        act_probs = reg_policy.compute_action_probs(obs=new_obs)
        self.assert_act_probs(act_probs, expected_shape=(10, 16))

    def test_http_policy(self):
        # create a fake HTTP server
        class DummyHttpPolicyRequestHandler(BaseHTTPRequestHandler):
            """HTTPServer mock request handler."""

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length"), 0)
                raw_body = self.rfile.read(content_len)
                body = json.loads(raw_body)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                log_probs = np.log(
                    generate_action_probs(traj_length=len(body["obs"]), num_actions=3)
                )
                self.wfile.write(b'{"log_probs": ' + json.dumps(log_probs.tolist()).encode() + b"}")

            def log_request(self, code="-", size="-"):
                pass

        # start the server
        http_server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 8082), DummyHttpPolicyRequestHandler  # noqa
        )
        with http_server:
            server_thread = threading.Thread(target=http_server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            # define the HTTP policy that will use the server
            http_policy = HttpPolicy(
                host="localhost",
                port=8082,
                path="/agent",
                request_payload_fun=lambda obs: {"obs": obs.tolist(), "eps_id": 123},
                response_payload_fun=lambda response: np.array(response.json()["log_probs"]),
                batch_size=1,
            )

            # check if the policy returns valid log-likelihoods
            remote_log_probs = http_policy.log_likelihoods(obs=np.random.rand(10, 5))
            http_server.shutdown()

        self.assert_log_probs(remote_log_probs, expected_shape=(10, 3))

    def test_compute_action_probs(self):
        rnd_policy = RandomPolicy(num_actions=3)
        act_probs = rnd_policy.compute_action_probs(obs=np.random.rand(10, 5))
        self.assert_act_probs(act_probs, expected_shape=(10, 3))

    def test_select_action(self):
        rnd_policy = RandomPolicy(num_actions=3)
        actions = rnd_policy.select_action(obs=np.random.rand(10, 5))
        self.assertIsInstance(actions, np.ndarray)
        self.assertEqual(len(actions), 10)
        self.assertTrue(all(0 <= action < 3 for action in actions))

    def assert_log_probs(self, log_probs: np.ndarray, expected_shape: tuple):
        self.assertTrue(np.all(np.isfinite(log_probs)))
        self.assertTrue(np.all(log_probs <= 0.0))
        self.assert_act_probs(np.exp(log_probs), expected_shape)

    def assert_act_probs(self, act_probs: np.ndarray, expected_shape: tuple):
        self.assertIsInstance(act_probs, np.ndarray)
        self.assertEqual(act_probs.shape, expected_shape)
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0, atol=1e-3))
