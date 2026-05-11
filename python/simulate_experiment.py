"""Run delayed-escape and stimulation-only simulations with a trained RNN.

Python outputs are saved as standardized MATLAB-compatible files in ``results/``.
MATLAB scripts in this repository read those files directly, so no dated or
machine-specific filenames are required.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from psychrnn.backend.models.basic import Basic

from rnn_task import (
    DelayedEscapeTask,
    StimulationTask,
    build_network_params,
    load_cluster_indices,
    run_test_batch,
    save_escape_results,
    select_device,
)


CONDITION_DEFAULTS: Dict[str, Dict] = {
    "main": {
        "door_delay": 50,
        "T": 2400,
        "door_signal_value": 1.0,
        "perturbation_value": None,
        "output": "results/rnn_main.mat",
    },
    "cs_only": {
        "door_delay": 50,
        "T": 2400,
        "door_signal_value": 0.0,
        "perturbation_value": None,
        "output": "results/rnn_cs_only.mat",
    },
    "inhibition": {
        "door_delay": 50,
        "T": 2400,
        "door_signal_value": 1.0,
        "perturbation_value": -1.0,
        "output": "results/rnn_inhibition.mat",
    },
    "interval_180s": {
        "door_delay": 1800,
        "T": 4000,
        "door_signal_value": 1.0,
        "perturbation_value": None,
        "output": "results/rnn_interval_180s.mat",
    },
    "stimulation": {
        "T": 310,
        "output": "results/rnn_stimulation.mat",
    },
    "stimulation_nbqx": {
        "T": 310,
        "output": "results/rnn_stimulation_nbqx.mat",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(CONDITION_DEFAULTS), required=True)
    parser.add_argument("--pretrained-weights", default="weights/rnn_main.npz")
    parser.add_argument("--output", default=None, help="Output .mat file. Defaults to a standardized path in results/.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--tau", type=int, default=10)
    parser.add_argument("--latency-jitter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    # Cluster arguments are needed only for inhibition and stimulation analyses.
    parser.add_argument("--cluster-file", default="results/fixed_tsne_clustering.mat")
    parser.add_argument("--cluster-label", type=int, default=2, help="Cluster label in clusterResult.idx.")
    parser.add_argument("--cluster-key", default=None, help="Optional vector variable if not using clusterResult.idx.")
    parser.add_argument("--input-gain", type=float, default=None, help="Input-channel 7 gain for selected cluster neurons.")
    parser.add_argument("--inhibition-start", type=int, default=900)
    parser.add_argument("--inhibition-end", type=int, default=950)

    # Stimulation-only settings. These occur before any CS or door cue and are
    # therefore independent of CS-present/door-open trial labels.
    parser.add_argument("--stimulation-start", type=int, default=200)
    parser.add_argument("--stimulation-end", type=int, default=210)
    parser.add_argument("--stimulation-value", type=float, default=1.0)
    parser.add_argument("--stimulation-gain", type=float, default=0.10)
    parser.add_argument("--inhibitory-scale", type=float, default=0.40)
    parser.add_argument("--nbqx-fraction", type=float, default=0.10)
    parser.add_argument("--nbqx-excitatory-scale", type=float, default=0.40)
    parser.add_argument("--nbqx-init-bin", type=int, default=30)
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> str:
    return args.output or CONDITION_DEFAULTS[args.condition]["output"]


def make_delayed_escape_task(args: argparse.Namespace) -> DelayedEscapeTask:
    defaults = CONDITION_DEFAULTS[args.condition]
    perturb_value = defaults["perturbation_value"]
    return DelayedEscapeTask(
        dt=args.dt,
        tau=args.tau,
        T=defaults["T"],
        N_batch=args.batch_size,
        door_delay=defaults["door_delay"],
        door_signal_value=defaults["door_signal_value"],
        perturbation_start=args.inhibition_start if perturb_value is not None else None,
        perturbation_end=args.inhibition_end if perturb_value is not None else None,
        perturbation_value=perturb_value if perturb_value is not None else -1.0,
        latency_jitter=args.latency_jitter,
    )


def make_stimulation_task(args: argparse.Namespace, T: Optional[int] = None, stimulation_value: Optional[float] = None) -> StimulationTask:
    defaults = CONDITION_DEFAULTS[args.condition]
    return StimulationTask(
        dt=args.dt,
        tau=args.tau,
        T=T if T is not None else defaults["T"],
        N_batch=args.batch_size,
        stimulation_start=args.stimulation_start,
        stimulation_end=args.stimulation_end,
        stimulation_value=args.stimulation_value if stimulation_value is None else stimulation_value,
    )


def load_model(task, args: argparse.Namespace, name: str) -> Basic:
    params = build_network_params(task, name=name)
    params["load_weights_path"] = args.pretrained_weights
    return Basic(params)


def reload_model_from_weights(model: Basic, task, weights: Dict[str, np.ndarray], temp_path: str, name: str) -> Basic:
    os.makedirs(os.path.dirname(temp_path) or ".", exist_ok=True)
    np.savez(temp_path, **weights)
    model.destruct()
    params = build_network_params(task, name=name)
    params["load_weights_path"] = temp_path
    return Basic(params)


def apply_cluster_input(model: Basic, task, cluster_indices: np.ndarray, gain: float, temp_path: str, name: str) -> Tuple[Basic, str]:
    """Route perturbation/stimulation input channel 7 to selected neurons."""
    weights = model.get_weights()
    W_in = weights["W_in"]
    W_in[:, 7] = 0.0
    W_in[cluster_indices, 7] = gain
    weights["W_in"] = W_in
    return reload_model_from_weights(model, task, weights, temp_path, name), temp_path


def restrict_to_cluster_recurrence(
    model: Basic,
    task,
    cluster_indices: np.ndarray,
    inhibitory_scale: float,
    temp_path: str,
    name: str,
) -> Tuple[Basic, str]:
    """Keep only within-cluster recurrence and weaken inhibitory columns."""
    weights = model.get_weights()
    W_rec = weights["W_rec"].copy()
    n_rec = W_rec.shape[0]
    n_excitatory = int(0.8 * n_rec)
    W_rec[:, n_excitatory:] *= inhibitory_scale
    mask = np.zeros_like(W_rec)
    mask[np.ix_(cluster_indices, cluster_indices)] = 1.0
    weights["W_rec"] = W_rec * mask
    return reload_model_from_weights(model, task, weights, temp_path, name), temp_path


def load_cluster_for_condition(args: argparse.Namespace) -> np.ndarray:
    if args.condition not in {"inhibition", "stimulation", "stimulation_nbqx"}:
        return np.array([], dtype=int)
    return load_cluster_indices(args.cluster_file, cluster_label=args.cluster_label, key=args.cluster_key)


def run_standard_condition(args: argparse.Namespace, cluster_indices: np.ndarray) -> None:
    task = make_delayed_escape_task(args)
    model = load_model(task, args, name=args.condition)
    temp_paths = []

    if args.condition == "inhibition":
        gain = args.input_gain if args.input_gain is not None else 0.15
        temp_path = os.path.join("results", f"temp_input_cluster_{args.cluster_label}.npz")
        model, temp_path = apply_cluster_input(model, task, cluster_indices, gain, temp_path, name=args.condition)
        temp_paths.append(temp_path)

    _, _, _, trial_params, model_output, model_state = run_test_batch(model, task)
    save_escape_results(
        default_output_path(args),
        model,
        model_output,
        model_state,
        trial_params,
        metadata={
            "condition": args.condition,
            "cluster_label": args.cluster_label if cluster_indices.size else np.nan,
        },
    )
    for path in temp_paths:
        if os.path.exists(path):
            os.remove(path)
    model.destruct()


def run_stimulation_condition(args: argparse.Namespace, cluster_indices: np.ndarray) -> None:
    task = make_stimulation_task(args)
    model = load_model(task, args, name=args.condition)
    temp_paths = []

    recurrence_path = os.path.join("results", f"temp_recurrent_cluster_{args.cluster_label}.npz")
    model, recurrence_path = restrict_to_cluster_recurrence(
        model, task, cluster_indices, args.inhibitory_scale, recurrence_path, name=f"{args.condition}_cluster"
    )
    temp_paths.append(recurrence_path)

    input_path = os.path.join("results", f"temp_stim_input_cluster_{args.cluster_label}.npz")
    model, input_path = apply_cluster_input(
        model, task, cluster_indices, args.stimulation_gain, input_path, name=f"{args.condition}_input"
    )
    temp_paths.append(input_path)

    _, _, _, trial_params, model_output, model_state = run_test_batch(model, task)
    save_escape_results(
        default_output_path(args),
        model,
        model_output,
        model_state,
        trial_params,
        metadata={"condition": args.condition, "cluster_label": args.cluster_label},
    )

    for path in temp_paths:
        if os.path.exists(path):
            os.remove(path)
    model.destruct()


def run_stimulation_nbqx_condition(args: argparse.Namespace, cluster_indices: np.ndarray) -> None:
    """Run front-segment stimulation, then continue from that state with weakened local excitation."""
    pre_task = make_stimulation_task(args, T=CONDITION_DEFAULTS[args.condition]["T"])
    model = load_model(pre_task, args, name="stimulation_nbqx_pre")
    temp_paths = []

    recurrence_path = os.path.join("results", f"temp_nbqx_recurrent_cluster_{args.cluster_label}.npz")
    model, recurrence_path = restrict_to_cluster_recurrence(
        model, pre_task, cluster_indices, args.inhibitory_scale, recurrence_path, name="stimulation_nbqx_recurrent"
    )
    temp_paths.append(recurrence_path)

    input_path = os.path.join("results", f"temp_nbqx_input_cluster_{args.cluster_label}.npz")
    model, input_path = apply_cluster_input(
        model, pre_task, cluster_indices, args.stimulation_gain, input_path, name="stimulation_nbqx_input"
    )
    temp_paths.append(input_path)

    _, _, _, pre_params, pre_output, pre_state = run_test_batch(model, pre_task)

    weights = model.get_weights()
    rng = np.random.default_rng(args.seed)
    n_rec = weights["W_rec"].shape[0]
    n_excitatory = int(0.8 * n_rec)
    n_selected = max(1, int(round(len(cluster_indices) * args.nbqx_fraction)))
    nbqx_targets = rng.choice(cluster_indices, size=n_selected, replace=False)
    weights["W_rec"][nbqx_targets, :n_excitatory] *= args.nbqx_excitatory_scale

    post_outputs, post_states, post_params = [], [], []
    post_task = StimulationTask(
        dt=args.dt,
        tau=args.tau,
        T=CONDITION_DEFAULTS[args.condition]["T"],
        N_batch=1,
        stimulation_start=10**9,  # no additional stimulation during continuation
        stimulation_end=10**9 + 1,
        stimulation_value=0.0,
    )
    post_temp = os.path.join("results", f"temp_nbqx_post_cluster_{args.cluster_label}.npz")
    temp_paths.append(post_temp)
    init_bin = min(args.nbqx_init_bin, pre_state.shape[1] - 1)

    for trial in range(pre_state.shape[0]):
        weights["init_state"] = pre_state[trial, init_bin, :]
        np.savez(post_temp, **weights)
        post_params_network = build_network_params(post_task, name=f"stimulation_nbqx_post_{trial}")
        post_params_network["load_weights_path"] = post_temp
        post_model = Basic(post_params_network)
        _, _, _, trial_param, output, state = run_test_batch(post_model, post_task)
        post_outputs.append(output)
        post_states.append(state)
        post_params.append(trial_param[0])
        post_model.destruct()

    post_output = np.concatenate(post_outputs, axis=0)
    post_state = np.concatenate(post_states, axis=0)

    # Save one continuous stimulation-only time series for MATLAB. The pre segment
    # is retained through init_bin, then the NBQX continuation is appended.
    combined_output = np.concatenate([pre_output[:, : init_bin + 1, :], post_output], axis=1)
    combined_state = np.concatenate([pre_state[:, : init_bin + 1, :], post_state], axis=1)
    save_escape_results(
        default_output_path(args),
        model,
        combined_output,
        combined_state,
        pre_params,
        metadata={
            "condition": args.condition,
            "cluster_label": args.cluster_label,
            "nbqx_targets_zero_indexed": nbqx_targets,
            "nbqx_init_bin": init_bin,
        },
    )

    for path in temp_paths:
        if os.path.exists(path):
            os.remove(path)
    model.destruct()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs("results", exist_ok=True)
    device = select_device()
    cluster_indices = load_cluster_for_condition(args)

    with tf.device(device):
        if args.condition == "stimulation":
            run_stimulation_condition(args, cluster_indices)
        elif args.condition == "stimulation_nbqx":
            run_stimulation_nbqx_condition(args, cluster_indices)
        else:
            run_standard_condition(args, cluster_indices)


if __name__ == "__main__":
    main()
