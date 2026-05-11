"""Shared PsychRNN task and utility functions for delayed-escape RNN simulations.

The public code intentionally contains only the delayed-escape task used in the
manuscript. Legacy non-manuscript task variants were removed so that the repository mirrors the analyses reported in the paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from psychrnn.backend.models.basic import Basic
from psychrnn.tasks.task import Task
from scipy.io import loadmat, savemat


@dataclass(frozen=True)
class Timing:
    """Timing constants in the PsychRNN time unit.

    With ``dt = 10``, a value of 700 corresponds to 70 one-second analysis bins.
    The default task is: 20 s CS, 5 s CS-offset-to-door interval, then door open.
    """

    cs_onset: int = 700
    cs_duration: int = 200
    mask_start: int = 100


@dataclass(frozen=True)
class NetworkDefaults:
    """Default network parameters used for training and test simulations."""

    n_rec: int = 100
    rec_noise: float = 0.05
    l1_in: float = 0.01
    l1_rec: float = 0.01
    l2_rec: float = 0.005
    l1_out: float = 0.01
    l2_firing_rate: float = 0.95
    dale_ratio: float = 0.8
    loss_function: str = "binary_cross_entropy"


class DelayedEscapeTask(Task):
    """Delayed-escape task used to train and test the RNN.

    Eight input channels and two output channels are retained for compatibility
    with the original saved PsychRNN weights. Only the following channels are
    used by the public delayed-escape model:

    * input 0: conditioned stimulus (CS)
    * input 1: door-open cue
    * input 3: task context
    * input 7: external perturbation, e.g. inhibition or stimulation

    The first output channel represents crossing probability. The second output
    channel is unused and is kept only for weight-file compatibility.
    """

    def __init__(
        self,
        dt: int,
        tau: int,
        T: int,
        N_batch: int,
        door_delay: int,
        door_signal_value: float = 1.0,
        perturbation_start: Optional[int] = None,
        perturbation_end: Optional[int] = None,
        perturbation_value: float = -1.0,
        latency_cs: int = 487,
        latency_absent: int = 1113,
        latency_jitter: int = 0,
        timing: Timing = Timing(),
    ):
        super().__init__(8, 2, dt, tau, T, N_batch)
        self.door_delay = int(door_delay)
        self.door_signal_value = float(door_signal_value)
        self.perturbation_start = perturbation_start
        self.perturbation_end = perturbation_end
        self.perturbation_value = float(perturbation_value)
        self.latency_cs = int(latency_cs)
        self.latency_absent = int(latency_absent)
        self.latency_jitter = int(latency_jitter)
        self.timing = timing
        self._batch_cs_present: List[int] = []

    def _make_balanced_cs_list(self) -> List[int]:
        labels = [0, 1] * (self.N_batch // 2)
        if len(labels) < self.N_batch:
            labels.append(int(np.random.choice([0, 1])))
        np.random.shuffle(labels)
        return labels

    def generate_trial_params(self, batch: int, trial: int) -> Dict[str, int]:
        if trial == 0 or not self._batch_cs_present:
            self._batch_cs_present = self._make_balanced_cs_list()

        cs_present = int(self._batch_cs_present[trial])
        if self.latency_jitter > 0:
            jitter = int(np.random.randint(-self.latency_jitter, self.latency_jitter + 1))
        else:
            jitter = 0

        return {
            "task_type": "delayed_escape",
            "CS_present": cs_present,
            "CS_onset": self.timing.cs_onset,
            "delay": self.door_delay,
            "latency": (self.latency_cs if cs_present else self.latency_absent) + jitter,
        }

    def trial_function(self, time: int, params: Dict[str, int]):
        x_t = np.zeros(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        cs_on = params["CS_onset"]
        cs_off = cs_on + self.timing.cs_duration
        door_on = cs_off + int(params["delay"])
        latency_on = door_on + int(params["latency"])

        x_t[3] = 1.0  # task-context channel
        if params["CS_present"] and cs_on <= time < cs_off:
            x_t[0] = 2.0
        if door_on <= time:
            x_t[1] = self.door_signal_value
        if latency_on <= time:
            y_t[0] = 1.0

        # Do not penalize output before the task-relevant response window.
        if cs_on <= time < door_on or time < self.timing.mask_start:
            mask_t[:] = 0.0

        if self.perturbation_start is not None and self.perturbation_end is not None:
            if self.perturbation_start <= time < self.perturbation_end:
                x_t[7] = self.perturbation_value

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """PsychRNN accuracy based on thresholded crossing output."""
        threshold = 0.5
        correct_binary = (correct_output[:, :, 0] >= threshold).astype(np.int32)
        test_binary = (test_output[:, :, 0] >= threshold).astype(np.int32)
        mask = output_mask[:, :, 0]
        correct_masked = correct_binary * mask
        test_masked = test_binary * mask
        matches = (correct_masked == 1) & (test_masked == 1)
        numerator = np.sum(matches)
        denominator = np.sum(test_masked) + np.sum(correct_masked) - numerator
        return 0.0 if denominator == 0 else numerator / denominator


class StimulationTask(Task):
    """Short stimulation-only test task.

    This task is independent of CS and door cues. It uses only the beginning of a
    simulation window, applies external drive through input channel 7, and stores
    the result with the same variable names as delayed-escape simulations so that
    MATLAB analysis can load it directly.
    """

    def __init__(
        self,
        dt: int,
        tau: int,
        T: int,
        N_batch: int,
        stimulation_start: int = 200,
        stimulation_end: int = 210,
        stimulation_value: float = 1.0,
    ):
        super().__init__(8, 2, dt, tau, T, N_batch)
        self.stimulation_start = int(stimulation_start)
        self.stimulation_end = int(stimulation_end)
        self.stimulation_value = float(stimulation_value)

    def generate_trial_params(self, batch: int, trial: int) -> Dict[str, int]:
        return {
            "task_type": "stimulation_only",
            "CS_present": 0,
            "CS_onset": 0,
            "delay": 0,
            "latency": 0,
            "stimulation_start": self.stimulation_start,
            "stimulation_end": self.stimulation_end,
        }

    def trial_function(self, time: int, params: Dict[str, int]):
        x_t = np.zeros(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)
        x_t[3] = 1.0  # keep the same context channel used by the trained model
        if self.stimulation_start <= time < self.stimulation_end:
            x_t[7] = self.stimulation_value
        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        return 0.0


def select_device() -> str:
    """Return the GPU device if available, otherwise CPU."""
    device = "/gpu:0" if tf.config.experimental.list_physical_devices("GPU") else "/cpu:0"
    print(f"Using device: {device}")
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU")))
    return device


def build_network_params(task: Task, name: str, defaults: NetworkDefaults = NetworkDefaults()) -> Dict:
    """Create PsychRNN network parameters matching the manuscript model."""
    params = task.get_task_params()
    params.update(
        {
            "name": name,
            "N_rec": defaults.n_rec,
            "rec_noise": defaults.rec_noise,
            "L1_in": defaults.l1_in,
            "L1_rec": defaults.l1_rec,
            "L2_rec": defaults.l2_rec,
            "L1_out": defaults.l1_out,
            "L2_firing_rate": defaults.l2_firing_rate,
            "dale_ratio": defaults.dale_ratio,
            "W_in_train": False,
            "b_in_train": False,
            "loss_function": defaults.loss_function,
        }
    )
    return params


def initialise_input_weights(weights: Dict[str, np.ndarray], dale_ratio: float, low: float = 0.0, high: float = 0.15):
    """Initialize input weights and zero input rows for inhibitory units."""
    W_in = np.random.uniform(low=low, high=high, size=weights["W_in"].shape)
    n_rec = W_in.shape[0]
    n_excitatory = int(dale_ratio * n_rec)
    W_in[n_excitatory:, :] = 0.0
    weights["W_in"] = W_in
    return weights


def _extract_mat_struct_field(mat_struct, field: str):
    """Extract a field from a scipy-loaded MATLAB struct."""
    arr = np.asarray(mat_struct)
    if arr.dtype.names and field in arr.dtype.names:
        return arr[field][0, 0]
    raise KeyError(field)


def load_cluster_indices(cluster_file: str, cluster_label: int = 2, key: Optional[str] = None) -> np.ndarray:
    """Load cluster neuron indices and return zero-indexed Python indices.

    Preferred input is the MATLAB output ``results/fixed_tsne_clustering.mat``,
    which contains ``clusterResult.idx``. For compatibility, a file containing a
    one-indexed vector variable can also be supplied via ``key``.
    """
    mat = loadmat(cluster_file)
    if "clusterResult" in mat:
        idx = np.asarray(_extract_mat_struct_field(mat["clusterResult"], "idx")).ravel().astype(int)
        return np.where(idx == int(cluster_label))[0]

    if key is None:
        candidates = [k for k in mat if not k.startswith("__")]
        if len(candidates) != 1:
            raise ValueError(f"Provide --cluster-key. Found variables: {candidates}")
        key = candidates[0]
    indices = np.asarray(mat[key]).ravel().astype(int)
    if indices.size and indices.min() >= 1:
        indices = indices - 1
    return indices


def save_escape_results(path: str, model: Basic, model_output, model_state, trial_params, metadata: Optional[Dict] = None):
    """Save model outputs in the MATLAB-compatible structure used downstream."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    weights = model.get_weights()
    result = {
        "W_in": weights["W_in"],
        "W_out": weights["W_out"],
        "W_rec": weights["W_rec"],
        "hidden_activity_escape": model_state,
        "model_output_escape": model_output,
        "trial_params_escape": trial_params,
    }
    if metadata:
        result["metadata"] = metadata
    savemat(path, result)
    print(f"Saved {path}")


def run_test_batch(model: Basic, task: Task):
    """Run a PsychRNN test batch and return inputs, targets, masks, params, output, state."""
    x_test, target_output, mask, trial_params = task.get_trial_batch()
    model_output, model_state = model.test(x_test)
    return x_test, target_output, mask, trial_params, model_output, model_state
