from hopes.data.batch_utils import *
from hopes.ope.q_models import RTGQModelHGBoost


def prepare_behavior_inputs_from_df(
    df_subset: pd.DataFrame,
    *,
    cols: list[str],
    steps_per_episode: int,
    idx_zone_temp: int,
    idx_fract_time: int,
    target_start_occ: float,
    htg_setpoint: float,
    clg_setpoint: float,
    temp_margin: float,
    reward_col: str = "reward",
) -> dict | None:
    # episode_dict creation + cleaning
    episode_dict = build_episode_dict(df_subset, cols=cols)
    n0 = len(episode_dict)

    episode_dict = drop_first_step_in_each_episode(episode_dict, n_drop=1)
    episode_dict, _ = filter_episodes_by_length(episode_dict, steps_per_episode=steps_per_episode)
    n1 = len(episode_dict)

    episode_dict, _ = filter_episodes_by_temp_condition(
        episode_dict,
        idx_zone_temp=idx_zone_temp,
        htg_setpoint=htg_setpoint,
    )
    n2 = len(episode_dict)

    # For DEBUG
    if n2 == 0:
        print(f"[SKIP] episodes: start={n0}, after_len={n1}, after_temp={n2}")
        return None

    # logged probs
    episode_dict = add_logged_policy_probs(episode_dict)

    # sticky + reward + scaling
    episode_dict = add_sticky_action(episode_dict)
    episode_dict = add_zone_temp_and_reward(
        episode_dict,
        idx_zone_temp=idx_zone_temp,
        idx_fract_time=idx_fract_time,
        occupied_period_start=target_start_occ,
        clg_setpoint=clg_setpoint,
        htg_setpoint=htg_setpoint,
        margin=temp_margin,
    )

    # if empty, bail
    if len(episode_dict) == 0:
        return None

    # scale rewards (can still fail if reward missing everywhere)
    try:
        episode_dict, _, _ = global_minmax_scale_reward(
            episode_dict, reward_col="reward", verbose=False
        )
    except ValueError:
        return None

    # episode -> batches
    episode_seqs = build_episode_sequences(episode_dict)
    (
        obs_batches,
        act_batches,
        sticky_act_batches,
        rew_batches,
        probs_batches,
        p_taken_batches,
        argmax_batches,
        episode_ids,
    ) = build_numpy_batches(episode_seqs)

    # flatten
    obs_flat, act_flat, sticky_act_flat, rew_flat, p_b_taken_flat = flatten_episode_batches(
        obs_batches=obs_batches,
        act_batches=act_batches,
        sticky_act_batches=sticky_act_batches,
        rew_batches=rew_batches,
        p_taken_batches=p_taken_batches,
        verbose=False,
    )

    # Q model (DM/DR)
    q_model = RTGQModelHGBoost(steps_per_episode=steps_per_episode, random_state=0)
    Q0, Q1, _ = q_model.fit_predict_q0_q1(
        obs_flat=obs_flat,
        act_flat=act_flat,
        rew_flat=rew_flat,
        return_model=False,
    )

    return dict(
        obs_batches=obs_batches,
        act_batches=act_batches,
        probs_batches=probs_batches,
        obs_flat=obs_flat,
        act_flat=act_flat,
        sticky_act_flat=sticky_act_flat,
        rew_flat=rew_flat,
        p_b_taken_flat=p_b_taken_flat,
        Q0=Q0,
        Q1=Q1,
        steps_per_episode=steps_per_episode,
        n_episodes=len(obs_batches),
    )
