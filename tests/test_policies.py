import http
import json
import threading
import unittest

import numpy as np

from hopes.policy.policies import HttpPolicy, RandomPolicy, RegressionBasedPolicy
from tests.action_probs_utils import generate_action_probs


class TestPolicies(unittest.TestCase):
    def test_rnd_policy(self):
        rnd_policy = RandomPolicy(num_actions=3)

        log_probs = rnd_policy.log_likelihoods(obs=np.random.rand(10, 5))
        self.assertIsInstance(log_probs, np.ndarray)
        self.assertEqual(log_probs.shape, (10, 3))
        self.assertTrue(np.all(log_probs <= 0.0))
        self.assertTrue(np.all(log_probs >= -np.inf))

        act_probs = np.exp(log_probs)
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))

    def test_regression_policy(self):
        # generate a random dataset of (obs, act) for target policy
        num_actions = 3
        num_obs = 5
        num_samples = 100
        obs = np.random.rand(num_samples, num_obs)
        act = np.random.randint(num_actions, size=num_samples)

        # create and fit a regression-based policy
        reg_policy = RegressionBasedPolicy(obs=obs, act=act, regression_model="logistic")
        reg_policy.fit()

        # check if the policy returns valid log-likelihoods
        new_obs = np.random.rand(10, num_obs)
        act_probs = reg_policy.compute_action_probs(obs=new_obs)

        self.assertIsInstance(act_probs, np.ndarray)
        self.assertEqual(act_probs.shape, (10, 3))
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))

    def test_http_policy(self):
        # create a fake HTTP server
        class DummyHttpPolicyRequestHandler(http.server.BaseHTTPRequestHandler):
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
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 8082), DummyHttpPolicyRequestHandler  # noqa
        )
        with server:
            server_thread = threading.Thread(target=server.serve_forever)
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
            server.shutdown()

        self.assertIsInstance(remote_log_probs, np.ndarray)
        self.assertEqual(remote_log_probs.shape, (10, 3))
        self.assertTrue(np.all(remote_log_probs <= 0.0))
        self.assertTrue(np.all(remote_log_probs >= -np.inf))
        act_probs = np.exp(remote_log_probs)
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))

    def test_compute_action_probs(self):
        rnd_policy = RandomPolicy(num_actions=3)
        act_probs = rnd_policy.compute_action_probs(obs=np.random.rand(10, 5))

        self.assertIsInstance(act_probs, np.ndarray)
        self.assertEqual(act_probs.shape, (10, 3))
        self.assertTrue(np.all(act_probs >= 0.0))
        self.assertTrue(np.all(act_probs <= 1.0))
        self.assertTrue(np.allclose(act_probs.sum(axis=1), 1.0))
