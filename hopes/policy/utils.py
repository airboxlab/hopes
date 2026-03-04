import logging
import tempfile
from pathlib import Path
from re import Pattern

import boto3
import numpy as np
import onnx
from onnx import numpy_helper

s3 = boto3.client("s3")


def log_probs_for_deterministic_policy(
    actions: np.ndarray, actions_bins: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """Compute the log probabilities of a given set of actions, assuming a given deterministic
    policy.

    We assign a probability of ~1 to the action returned by the function and an almost zero
    probability to all other actions (note: sum of log probs must be 1)

    :param actions: the actions for which to compute the log probabilities.
    :param actions_bins: the set of possible actions.
    :param epsilon: the small value to use for the probabilities of the other actions.
    """
    assert np.all(np.isin(actions, actions_bins)), "Some actions are not in the action bins."

    # get index of each action in actions_bins
    act_idx = np.searchsorted(actions_bins, actions)

    # create the log probabilities
    unlikely_p = epsilon / len(actions_bins)
    act_probs = np.where(np.eye(len(actions_bins)) == 0, unlikely_p, 1.0 - epsilon + unlikely_p)
    return np.log([act_probs[a] for a in act_idx])


def bin_actions(actions: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Bin the given actions into the given bins."""
    return np.array([min(bins, key=lambda x: abs(x - ra)) for ra in actions])


def piecewise_linear(x, left_cp, right_cp, slope, y0, y1) -> np.ndarray:
    r"""Define a piecewise linear function with 3 segments, such as:

     y0 --- \ (1)
             \ slope
              \
           (2) \ --- y1

    (1) left_cp (2) right_cp
    Note: the slope is not necessarily negative, the 2nd segment function can be increasing or decreasing.

    :param x: the input variable.
    :param left_cp: the left change point.
    :param right_cp: the right change point.
    :param slope: the slope of the linear segment.
    :param y0: the base value of the left segment.
    :param y1: the base value of the right segment.
    """
    # define the conditions for each segment
    conditions = [x < left_cp, (x >= left_cp) & (x <= right_cp), x > right_cp]
    # first segment is flat until lcp
    # second segment is linear between lcp and rcp
    # third segment is flat after rcp
    funcs = [
        lambda _: y0,
        lambda v: slope * (v - left_cp) + y0,
        lambda _: y1,
    ]
    return np.piecewise(x, conditions, funcs)


def load_latest_model_onnx(bucket: str, prefix: str, checkpoint_model_regex: Pattern):
    """Find and load the model.onnx corresponding to the highest checkpoint number under the given
    S3 prefix.

    Returns:
        model_bytes (bytes): ONNX model content
        checkpoint_num (int): highest checkpoint number
        model_key (str): full S3 key of the selected model
    """
    paginator = s3.get_paginator("list_objects_v2")

    models_by_checkpoint = {}

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            match = checkpoint_model_regex.search(key)
            if match:
                checkpoint_num = int(match.group(1))
                models_by_checkpoint.setdefault(checkpoint_num, []).append(
                    {
                        "Key": key,
                        "LastModified": obj["LastModified"],
                    }
                )

    if not models_by_checkpoint:
        raise FileNotFoundError(f"No model.onnx found under prefix: s3://{bucket}/{prefix}")

    # Select highest checkpoint
    latest_checkpoint = max(models_by_checkpoint.keys())

    # If multiple models exist for the same checkpoint, take the most recent one
    latest_obj = max(models_by_checkpoint[latest_checkpoint], key=lambda x: x["LastModified"])

    model_key = latest_obj["Key"]

    response = s3.get_object(Bucket=bucket, Key=model_key)
    model_bytes = response["Body"].read()

    return model_bytes, latest_checkpoint, model_key


def remove_onnx_model_initializer(model: onnx.ModelProto, name: str) -> bool:
    """Remove initializer with given name from ONNX model and add it as model input if not already
    present."""
    for i, init in enumerate(model.graph.initializer):
        if name in init.name:
            old_value = numpy_helper.to_array(init)
            logging.info(
                f"[ONNX model prep] Found initializer '{name}' with shape {old_value.shape}, dtype {old_value.dtype} and value {old_value}"
            )

            # Move initializer to model input so it can be provided at runtime.
            if not any(inp.name == init.name for inp in model.graph.input):
                value_info = onnx.helper.make_tensor_value_info(
                    init.name, init.data_type, list(old_value.shape)
                )
                model.graph.input.append(value_info)

            del model.graph.initializer[i]
            logging.info(f"[ONNX model prep] Converted initializer '{name}' to model input")
            return True
    return False


def prepare_onnx_model(model_in: str, model_out: str) -> None:
    """Utility to prepare ONNX model by zeroing out specific initializers."""
    model = onnx.load(model_in)
    targets = ["is_exploring"]
    for name in targets:
        if remove_onnx_model_initializer(model, name):
            logging.info(f"[ONNX model prep] initializer '{name}' now expected as model input")
        else:
            logging.info(f"[ONNX model prep] initializer '{name}' not found in the model")

    onnx.save(model, model_out)


def write_onnx_bytes(model_bytes: bytes, name: str) -> str:
    p = Path(tempfile.gettempdir()) / f"{name}.onnx"
    p.write_bytes(model_bytes)
    return str(p)
