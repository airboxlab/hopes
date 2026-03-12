import unittest

import numpy as np

from hopes.data.pre_processing import (
    apply_stickiness_correction,
    build_stepwise_importance_ratios,
    build_stepwise_importance_ratios_with_stickiness,
    extract_logged_action_probabilities,
)


class TestPreProcessing(unittest.TestCase):
    def test_extract_logged_action_probabilities(self):
        action_probabilities = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        logged_actions = np.array([0, 1, 0, 1], dtype=np.int64)

        result = extract_logged_action_probabilities(
            action_probabilities=action_probabilities,
            logged_actions=logged_actions,
        )

        expected = np.array([0.8, 0.6, 0.7, 0.9], dtype=np.float32)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.allclose(result, expected))

    def test_extract_logged_action_probabilities_raises_for_invalid_ndim(self):
        action_probabilities = np.array([0.8, 0.2], dtype=np.float32)
        logged_actions = np.array([0], dtype=np.int64)

        with self.assertRaises(ValueError):
            extract_logged_action_probabilities(
                action_probabilities=action_probabilities,
                logged_actions=logged_actions,
            )

    def test_extract_logged_action_probabilities_raises_for_length_mismatch(self):
        action_probabilities = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )
        logged_actions = np.array([0], dtype=np.int64)

        with self.assertRaises(ValueError):
            extract_logged_action_probabilities(
                action_probabilities=action_probabilities,
                logged_actions=logged_actions,
            )

    def test_extract_logged_action_probabilities_raises_for_invalid_action_index(self):
        action_probabilities = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )
        logged_actions = np.array([0, 2], dtype=np.int64)

        with self.assertRaises(ValueError):
            extract_logged_action_probabilities(
                action_probabilities=action_probabilities,
                logged_actions=logged_actions,
            )

    def test_build_stepwise_importance_ratios(self):
        target = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        behavior = np.array(
            [
                [0.5, 0.5],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )
        logged_actions = np.array([0, 1, 0, 1], dtype=np.int64)

        rho = build_stepwise_importance_ratios(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            logged_actions=logged_actions,
        )

        expected = np.array(
            [0.8 / 0.5, 0.6 / 0.8, 0.7 / 0.5, 0.9 / 0.6],
            dtype=np.float32,
        )

        self.assertIsInstance(rho, np.ndarray)
        self.assertEqual(rho.shape, (4,))
        self.assertTrue(np.allclose(rho, expected))

    def test_build_stepwise_importance_ratios_with_clip(self):
        target = np.array(
            [
                [0.9, 0.1],
                [0.9, 0.1],
            ],
            dtype=np.float32,
        )
        behavior = np.array(
            [
                [0.1, 0.9],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        logged_actions = np.array([0, 0], dtype=np.int64)

        rho = build_stepwise_importance_ratios(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            logged_actions=logged_actions,
            clip=2.0,
        )

        expected = np.array([2.0, 2.0], dtype=np.float32)

        self.assertTrue(np.allclose(rho, expected))

    def test_build_stepwise_importance_ratios_raises_for_invalid_eps(self):
        target = np.array([[0.8, 0.2]], dtype=np.float32)
        behavior = np.array([[0.5, 0.5]], dtype=np.float32)
        logged_actions = np.array([0], dtype=np.int64)

        with self.assertRaises(ValueError):
            build_stepwise_importance_ratios(
                target_policy_action_probabilities=target,
                behavior_policy_action_probabilities=behavior,
                logged_actions=logged_actions,
                eps=0.0,
            )

    def test_build_stepwise_importance_ratios_raises_for_invalid_clip(self):
        target = np.array([[0.8, 0.2]], dtype=np.float32)
        behavior = np.array([[0.5, 0.5]], dtype=np.float32)
        logged_actions = np.array([0], dtype=np.int64)

        with self.assertRaises(ValueError):
            build_stepwise_importance_ratios(
                target_policy_action_probabilities=target,
                behavior_policy_action_probabilities=behavior,
                logged_actions=logged_actions,
                clip=0.5,
            )

    def test_apply_stickiness_correction_no_switch_effect_if_no_sticky_actions(self):
        importance_ratios = np.array([1.2, 0.8, 1.5, 0.7], dtype=np.float32)
        sticky_actions = np.array([0, 0, 0, 0], dtype=np.int64)

        corrected = apply_stickiness_correction(
            importance_ratios=importance_ratios,
            sticky_actions=sticky_actions,
            steps_per_episode=2,
            value_after_switch=1.0,
        )

        self.assertTrue(np.allclose(corrected, importance_ratios))

    def test_apply_stickiness_correction_shape(self):
        importance_ratios = np.array([1.1, 0.9, 1.3, 0.7], dtype=np.float32)
        sticky_actions = np.array([0, 1, 0, 0], dtype=np.int64)

        corrected = apply_stickiness_correction(
            importance_ratios=importance_ratios,
            sticky_actions=sticky_actions,
            steps_per_episode=2,
            value_after_switch=1.0,
        )

        self.assertIsInstance(corrected, np.ndarray)
        self.assertEqual(corrected.shape, (4,))

    def test_apply_stickiness_correction_raises_for_invalid_steps(self):
        importance_ratios = np.array([1.2, 0.8], dtype=np.float32)
        sticky_actions = np.array([0, 1], dtype=np.int64)

        with self.assertRaises(ValueError):
            apply_stickiness_correction(
                importance_ratios=importance_ratios,
                sticky_actions=sticky_actions,
                steps_per_episode=0,
                value_after_switch=1.0,
            )

    def test_apply_stickiness_correction_raises_for_length_mismatch(self):
        importance_ratios = np.array([1.2, 0.8, 1.5], dtype=np.float32)
        sticky_actions = np.array([0, 1], dtype=np.int64)

        with self.assertRaises(ValueError):
            apply_stickiness_correction(
                importance_ratios=importance_ratios,
                sticky_actions=sticky_actions,
                steps_per_episode=1,
                value_after_switch=1.0,
            )

    def test_apply_stickiness_correction_raises_if_not_divisible_by_steps(self):
        importance_ratios = np.array([1.2, 0.8, 1.5], dtype=np.float32)
        sticky_actions = np.array([0, 1, 0], dtype=np.int64)

        with self.assertRaises(ValueError):
            apply_stickiness_correction(
                importance_ratios=importance_ratios,
                sticky_actions=sticky_actions,
                steps_per_episode=2,
                value_after_switch=1.0,
            )

    def test_build_stepwise_importance_ratios_with_stickiness_no_stickiness(self):
        target = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        behavior = np.array(
            [
                [0.5, 0.5],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )
        logged_actions = np.array([0, 1, 0, 1], dtype=np.int64)

        rho = build_stepwise_importance_ratios_with_stickiness(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            logged_actions=logged_actions,
            steps_per_episode=2,
            apply_stickiness=False,
        )

        expected = np.array(
            [0.8 / 0.5, 0.6 / 0.8, 0.7 / 0.5, 0.9 / 0.6],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(rho, expected))

    def test_build_stepwise_importance_ratios_with_stickiness_raises_if_missing_sticky_actions(
        self,
    ):
        target = np.array([[0.8, 0.2]], dtype=np.float32)
        behavior = np.array([[0.5, 0.5]], dtype=np.float32)
        logged_actions = np.array([0], dtype=np.int64)

        with self.assertRaises(ValueError):
            build_stepwise_importance_ratios_with_stickiness(
                target_policy_action_probabilities=target,
                behavior_policy_action_probabilities=behavior,
                logged_actions=logged_actions,
                steps_per_episode=1,
                apply_stickiness=True,
                sticky_actions=None,
                value_after_switch=1.0,
            )

    def test_build_stepwise_importance_ratios_with_stickiness_raises_if_missing_value_after_switch(
        self,
    ):
        target = np.array([[0.8, 0.2]], dtype=np.float32)
        behavior = np.array([[0.5, 0.5]], dtype=np.float32)
        logged_actions = np.array([0], dtype=np.int64)
        sticky_actions = np.array([1], dtype=np.int64)

        with self.assertRaises(ValueError):
            build_stepwise_importance_ratios_with_stickiness(
                target_policy_action_probabilities=target,
                behavior_policy_action_probabilities=behavior,
                logged_actions=logged_actions,
                steps_per_episode=1,
                apply_stickiness=True,
                sticky_actions=sticky_actions,
                value_after_switch=None,
            )

    def test_build_stepwise_importance_ratios_with_stickiness_returns_expected_shape(self):
        target = np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        behavior = np.array(
            [
                [0.5, 0.5],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )
        logged_actions = np.array([0, 1, 0, 1], dtype=np.int64)
        sticky_actions = np.array([0, 1, 0, 0], dtype=np.int64)

        rho = build_stepwise_importance_ratios_with_stickiness(
            target_policy_action_probabilities=target,
            behavior_policy_action_probabilities=behavior,
            logged_actions=logged_actions,
            steps_per_episode=2,
            apply_stickiness=True,
            sticky_actions=sticky_actions,
            value_after_switch=1.0,
        )

        self.assertIsInstance(rho, np.ndarray)
        self.assertEqual(rho.shape, (4,))
        self.assertTrue(np.all(np.isfinite(rho)))
