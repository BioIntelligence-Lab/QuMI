# Expanding the Horizon: Enabling Hybrid Quantum Transfer Learning for Long-Tailed Chest X-Ray Classification

Read our preprint at <https://arxiv.org/abs/2405.00156>.

Please open an issue if you encounter difficulties reproducing our results.

## Install

`git clone https://github.com/BioIntelligence-Lab/QuMI.git`

There are several environment files which can be installed from mamba:

- `environment.yml` for the dependencies to run the CXR-8, CXR-14, and CXR-19 QML models with Jax. You will also need to install `scikit-multilearn-ng` from the included git submodule.
- `environments/environment-{jax,pytorch,tensorflow}.yml` for the dependencies to run minimal Pytorch/Tensorflow/Jax variants. Environments are kept separate to avoid CUDA conflicts.

Ensure that the Nvidia, Pytorch, and Conda-Forge channels are configured before installing.

The `minimal` variants only specify top-level dependencies to ease automated installation of lower-level dependencies if they become incompatible.

You can also install dependencies manually:

Deep learning
- transformers
- jax
- jaxlib with cuda
- flax
- optax
- orbax

Image preprocessing
- datasets
- dm-pix

Quantum computing
- pennylane

Dataloader
- joblib
- safetensors
- diskcache
- pyzstd
- safetensors

Data science
- pandas
- scikit-multilearn-ng

Coding style
- ruff

## GPU memory patch

In case you have issues with running out of GPU memory, you may need to apply `fix_XLA_PYTHON_CLIENT_PREALLOCATE_parsing.patch` to `xla/python/xla_client.py`. This patch was not applied to the version of Jax used in our experiments.

## Benchmark tests

`jax_model.py`, `tensorflow_model.py`, and `pytorch_model.py` require the respective environments in `environments/`. After activating the environment, run the script to benchmark this Pennylane backend. Results are in `benchmark_times`

## Datasets

NIH-CXR is a publicly available dataset used for model training: <https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/223604149466>.

MIMIC-CXR is a private dataset requiring credentialed access and is used for external testing.

For both datasets, we use the MICCAI-2023 long tail labels. NIH CXR labels are already included in `PruneCXRlabels/`

Please edit the code to specify where you downloaded these datasets and label CSV files.

## Paths

Paths to datasets, cache dir, experiment dir can be provided in the command line. `config.yml` has higher priority.

## Build cache

```
python src/main.py cache --cache-dir ./.cache/
```

Uses DiskCache to store resized images as zstd-compressed safetensors.

Will take about 1 hour for NIH and 3 hours for MIMIC on a single thread (multiple threads will block each other).

When the cache is built, it's very fast to iterate over (~10x speedup).

### Preprocessing

1. Resize image to 256 on the shortest side
2. Center crop image to 256x256
3. Rescale and normalize channels to ImageNet values
4. Cache this image

### Augmentation

1. Load the image from cache
3. Random flip (p = 0.5)
2. Random rotate (theta = +/- 15)
4. Random crop to 224x224

## Experiments

Model checkpoints, flops, logs, and results are stored in the experiments dir. In this repository it's `ISVLSI/{parameters}/{hyperparameters}`

### Train: `train_models.sh`
- trains a single model on 5 random seeds
- edit the experiment dir and data dir in the script
- model variations (6 total)
  - classical/quantum model
  - number of labels (8/14/19)
  - gpu number

To run all models tested: run the script 6 times. Use a different GPU number for each model if training in parallel.

### Test: `test_models_nih.sh` and `test_models_mimic.sh`

- edit the experiment dir and data dir in the script
- will automatically analyze all 6 model variations

### Analyze data

`analysis.py`

## Helper scripts for development

- `clean-my-tmp` cleans `/tmp` files from Jax JIT
- `ruff.sh` makes verifying `ruff` changes easier

