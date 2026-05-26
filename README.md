# Computational mechanisms for temporal integration in the anterior claustrum

This repository contains cleaned reproduction code for the recurrent neural network (RNN) simulations and downstream population analyses associated with the eLife reviewed preprint:

**Kuenbae Sohn, Donghyeon Yoon, Junghwa Lee, and Sukwoo Choi. _Computational mechanisms for temporal integration in the anterior claustrum_. eLife Reviewed Preprint 109539.**  
Paper: <https://elifesciences.org/reviewed-preprints/109539>  
DOI: <https://doi.org/10.7554/eLife.109539.2>

Please cite the eLife paper when using this code or the associated deposited data.

## Scope of this repository

The code is organized as a final analysis workflow rather than as exploratory development scripts. It implements the delayed-escape RNN simulations and analyses reported in the paper, including population clustering, decoding, PCA trajectories, nonlinear trajectory prediction, PID/synergy analysis, cross-temporal decoding, biological GPFA trajectory visualization, and slice-stimulation analog simulations.

Non-manuscript task variants and exploratory code paths were removed. Population clustering uses only the t-SNE setting reported in the paper:

```text
perplexity = 24
exaggeration = 48
random seed = 1
NumDimensions = 3
```

No t-SNE parameter sweep is performed by the public workflow.

## Repository layout

```text
.
├── python/
│   ├── rnn_task.py             # Delayed-escape and stimulation-only task definitions
│   ├── train_rnn.py            # RNN training and main test-batch export
│   └── simulate_experiment.py  # CS-only, inhibition, 180-s interval, and stimulation-only exports
├── matlab/
│   ├── config_analysis.m       # Shared constants and standardized result paths
│   ├── run_01_population_clustering.m
│   ├── run_02_decoders_and_pid.m
│   ├── run_03_pca_trajectories.m
│   ├── run_04_stimulation_analysis.m
│   ├── run_05_visualize_rnn_trajectories.m
│   ├── run_06_nonlinear_trajectory_prediction.m
│   ├── run_07_cross_temporal_decoding.m
│   ├── run_08_gpfa_biological_trajectories.m
│   └── helpers/
├── results/                    # Standardized generated or downloaded outputs
├── weights/                    # Trained PsychRNN weights
├── data/README.md              # Notes for external deposited data
├── docs/
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── .gitignore
```

## License

The software code in this repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.

This license applies to the code in this repository. Large generated outputs, trained weights, and experimental/source data are distributed separately through the public data archive listed in the paper's Data Availability Statement and may be governed by the license or terms of that archive.

## Python environment

```bash
pip install -r requirements.txt
```

PsychRNN and TensorFlow version compatibility can be sensitive. The paper simulations used TensorFlow 2.1.0, so a clean virtual environment is recommended.

## Standard generated files

Python and MATLAB communicate through generic `.mat` files in `results/`:

```text
results/rnn_main.mat              # CS + door-opening and open-only trials
results/rnn_cs_only.mat           # CS-only and no-CS/no-door trials
results/rnn_interval_180s.mat     # 180-s CS-offset-to-door interval trials
results/rnn_inhibition.mat        # cluster-targeted inhibition trials
results/rnn_stimulation.mat       # stimulation-only front-segment simulation
results/rnn_stimulation_nbqx.mat  # stimulation-only simulation with local recurrent weakening
results/fixed_tsne_clustering.mat # fixed-parameter clustering result
```

These names are independent of local dates, machine paths, or exploratory run labels.

## Standard public/source-data files

The additional public-source-data analyses expect standardized data files in `results/`:

```text
results/rnn_activity_public.mat       # z-scored RNN activity for CS+Open, Open-only, and CS-only
results/rnn_perturbations_public.mat  # inhibition and 180-s interval activity
results/rnn_pid_synergy_public.mat    # PID/synergy matrices used for high/low synergy split
results/biological_gpfa_public.mat    # biological recording input for GPFA
results/fixed_tsne_clustering.mat     # single fixed t-SNE clustering result
```

These files should be deposited in a public archive or provided as a separate data bundle. They are intentionally not committed to GitHub because the `.mat` files are large.

## Workflow from generated RNN outputs

Train a delayed-escape RNN and export the main test batch:

```bash
python python/train_rnn.py \
  --runs 1 \
  --training-iters 1200000 \
  --output-dir results \
  --weights-dir weights \
  --output-prefix rnn_main
```

Generate CS-only/no-door trials:

```bash
python python/simulate_experiment.py \
  --condition cs_only \
  --pretrained-weights weights/rnn_main.npz
```

Run fixed-parameter population clustering:

```matlab
addpath(genpath('matlab'))
run('matlab/run_01_population_clustering.m')
```

Generate perturbation and stimulation simulations:

```bash
python python/simulate_experiment.py \
  --condition inhibition \
  --pretrained-weights weights/rnn_main.npz \
  --cluster-file results/fixed_tsne_clustering.mat \
  --cluster-label 2

python python/simulate_experiment.py \
  --condition interval_180s \
  --pretrained-weights weights/rnn_main.npz

python python/simulate_experiment.py \
  --condition stimulation \
  --pretrained-weights weights/rnn_main.npz \
  --cluster-file results/fixed_tsne_clustering.mat \
  --cluster-label 2

python python/simulate_experiment.py \
  --condition stimulation_nbqx \
  --pretrained-weights weights/rnn_main.npz \
  --cluster-file results/fixed_tsne_clustering.mat \
  --cluster-label 2
```

The stimulation conditions are independent of CS and door cues. They test only the early/front segment and inject external drive through input channel 7.

## MATLAB analyses

For analyses using generated outputs:

```matlab
addpath(genpath('matlab'))
run('matlab/run_02_decoders_and_pid.m')
run('matlab/run_03_pca_trajectories.m')
run('matlab/run_04_stimulation_analysis.m')
```

For analyses using the standardized public data files, place the files in `results/`, then run:

```matlab
addpath(genpath('matlab'))
run('matlab/run_05_visualize_rnn_trajectories.m')
run('matlab/run_06_nonlinear_trajectory_prediction.m')
run('matlab/run_07_cross_temporal_decoding.m')
run('matlab/run_08_gpfa_biological_trajectories.m')
```

`run_02_decoders_and_pid.m` requires `quickPID` for PID. `run_06_nonlinear_trajectory_prediction.m` uses `fitnet`. `run_08_gpfa_biological_trajectories.m` requires NeuralTraj. If a dependency is absent, the relevant script reports the missing dependency instead of silently failing.

## MATLAB requirements

The MATLAB scripts use the Statistics and Machine Learning Toolbox, including `tsne`, `kmeans`, `evalclusters`, `fitcdiscr`, and cross-validation utilities. Some optional analyses also require the Deep Learning Toolbox/Neural Network Toolbox and NeuralTraj.

## Data availability

Large `.mat` outputs and trained weights are not committed here. Deposit standardized generated outputs and model weights in the public archive selected for the paper, or regenerate them with the commands above. The public archive should contain the files listed in the Standard generated files and Standard public/source-data files sections, or equivalent files using the same variable structure.

## Citation

```text
Sohn K, Yoon D, Lee J, Choi S. Computational mechanisms for temporal integration in the anterior claustrum. eLife Reviewed Preprint 109539. doi:10.7554/eLife.109539.2
```

