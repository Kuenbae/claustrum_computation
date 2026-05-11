"""Train delayed-escape PsychRNN models and export MATLAB-compatible results."""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf
from psychrnn.backend.curriculum import Curriculum
from psychrnn.backend.models.basic import Basic
from scipy.io import savemat

from rnn_task import (
    DelayedEscapeTask,
    build_network_params,
    initialise_input_weights,
    run_test_batch,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="Number of independent training runs.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed; run index is added for repeated runs.")
    parser.add_argument("--training-iters", type=int, default=1_200_000, help="Training iterations per run.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training and testing batch size.")
    parser.add_argument("--dt", type=int, default=10, help="PsychRNN integration step.")
    parser.add_argument("--tau", type=int, default=10, help="PsychRNN time constant.")
    parser.add_argument("--T", type=int, default=2400, help="Trial duration in PsychRNN time units.")
    parser.add_argument("--latency-jitter", type=int, default=0, help="Optional target-latency jitter in PsychRNN time units.")
    parser.add_argument("--output-dir", default="results", help="Directory for exported .mat files.")
    parser.add_argument("--weights-dir", default="weights", help="Directory for exported PsychRNN weights.")
    parser.add_argument("--output-prefix", default="rnn_main", help="Prefix for exported run files.")
    return parser.parse_args()


def make_curriculum(dt: int, tau: int, T: int, batch_size: int, latency_jitter: int):
    """Create the four-step door-delay curriculum used for delayed escape."""
    delays = [0, 10, 25, 50]
    tasks = [
        DelayedEscapeTask(
            dt=dt,
            tau=tau,
            T=T,
            N_batch=batch_size,
            door_delay=delay,
            latency_jitter=latency_jitter,
        )
        for delay in delays
    ]
    thresholds = [0.5, 0.5, 0.5, 1.0]
    return tasks[-1], Curriculum(tasks, thresholds=thresholds)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    tf.config.optimizer.set_jit(True)
    device = select_device()

    for run_idx in range(1, args.runs + 1):
        np.random.seed(args.seed + run_idx)
        tf.random.set_seed(args.seed + run_idx)
        print(f"\n=== Training delayed-escape RNN run {run_idx}/{args.runs} ===")

        with tf.device(device):
            test_task, curriculum = make_curriculum(args.dt, args.tau, args.T, args.batch_size, args.latency_jitter)
            network_params = build_network_params(test_task, name=f"delayed_escape_run_{run_idx}")

            # Build once to obtain weight shapes, replace W_in, and then reload.
            model = Basic(network_params)
            weights = initialise_input_weights(model.get_weights(), dale_ratio=network_params["dale_ratio"])
            temp_weights = os.path.join(args.weights_dir, f"initialised_run_{run_idx:03d}.npz")
            np.savez(temp_weights, **weights)
            model.destruct()

            network_params["load_weights_path"] = temp_weights
            model = Basic(network_params)
            train_params = {
                "training_iters": args.training_iters,
                "curriculum": curriculum,
                "learning_rate": args.learning_rate,
            }
            losses, initial_time, train_time = model.train_curric(train_params)
            print(f"Run {run_idx} training completed in {train_time:.2f} seconds.")

            _, _, _, trial_params, model_output, hidden_activity = run_test_batch(model, test_task)
            weights = model.get_weights()
            suffix = "" if args.runs == 1 else f"_run_{run_idx:03d}"
            output_path = os.path.join(args.output_dir, f"{args.output_prefix}{suffix}.mat")
            savemat(
                output_path,
                {
                    "W_in": weights["W_in"],
                    "W_out": weights["W_out"],
                    "W_rec": weights["W_rec"],
                    "hidden_activity_escape": hidden_activity,
                    "model_output_escape": model_output,
                    "trial_params_escape": trial_params,
                    "losses": losses,
                    "metadata": {
                        "condition": "main",
                        "task": "delayed_escape",
                        "door_delay": 50,
                    },
                },
            )
            weights_path = os.path.join(args.weights_dir, f"{args.output_prefix}{suffix}.npz")
            np.savez(weights_path, **weights)
            model.save(os.path.join(args.weights_dir, f"{args.output_prefix}{suffix}"))
            print(f"Saved {output_path}")
            print(f"Saved {weights_path}")

            if os.path.exists(temp_weights):
                os.remove(temp_weights)
            model.destruct()


if __name__ == "__main__":
    main()
