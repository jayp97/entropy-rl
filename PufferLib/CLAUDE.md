# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PufferLib is a high-performance reinforcement learning library (v3.0.0). It provides environment compatibility wrappers, optimized parallel simulation, built-in game environments ("Ocean"), and a PPO training system that achieves 1M+ steps/second. Python 3.9-3.14, MIT licensed.

## Build & Install

```bash
# Install in editable mode (required: --no-build-isolation due to torch/numpy/Cython build deps)
pip install -e . --no-build-isolation

# Build with debug symbols and sanitizers
DEBUG=1 python setup.py build_ext --inplace --force

# Build without Ocean environments (skip Raylib/Box2D downloads)
NO_OCEAN=1 pip install -e . --no-build-isolation

# Build without training extensions (skip C++/CUDA)
NO_TRAIN=1 pip install -e . --no-build-isolation

# Build a single Ocean environment
python setup.py build_<env_name>
# e.g., python setup.py build_cartpole
```

CUDA extensions are auto-detected via `CUDA_HOME` or `ROCM_HOME`. The build downloads Raylib 5.5 and Box2D automatically.

## Running

```bash
# Training (CLI entry point)
puffer train <env_name> [args]
# Or: python -m pufferlib.pufferl train <env_name> [args]

# Evaluation
puffer eval <env_name> [args]

# Hyperparameter sweep
puffer sweep <env_name> [args]

# Distributed training
torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3
```

## Testing

```bash
pytest tests/
pytest tests/test_api.py          # single test file
```

Key test files: `test_api.py` (Gymnasium/PettingZoo/vectorization), `test_extensions.py` (C/CUDA), `test_puffernet.py` (models), `test_vector.py` (vectorization backends), `test_performance.py` (benchmarks).

## Architecture

### Core Pipeline

1. **PufferEnv** (`pufferlib/pufferlib.py`) — Base environment class. Requires `single_observation_space`, `single_action_space`, and `num_agents` set *before* `super().__init__()`. Uses in-place array updates for zero-copy efficiency via `set_buffers()`.

2. **Emulation** (`pufferlib/emulation.py`) — Wraps external Gymnasium/PettingZoo environments into the PufferEnv interface. `GymnasiumPufferEnv` and `PettingZooPufferEnv` handle space conversion and structured dtype flattening.

3. **Vectorization** (`pufferlib/vector.py`) — Three backends: `Serial`, `Multiprocessing`, `Ray`. Async API (`async_reset()`, `send()`, `recv()`) and sync API (`reset()`, `step()`). Manages shared observation/reward/terminal buffers across envs.

4. **Models** (`pufferlib/models.py`) — `Default` policy with encoder-decoder pattern. `LSTMWrapper` for recurrent policies. Supports Discrete, MultiDiscrete, and continuous (Box) action spaces. Forward split into `encode_observations` / `decode_actions` for LSTM compatibility.

5. **Training** (`pufferlib/pufferl.py`) — `PuffeRL` class orchestrates PPO training with `torch.distributed` support. CLI parses INI config files from `pufferlib/config/`. C/CUDA advantage kernel via `pufferlib._C`.

6. **Sweep** (`pufferlib/sweep.py`) — Bayesian optimization with Gaussian processes (GPyTorch). Parameter spaces: Linear, Pow2, Log, Logit. Neptune/W&B integration.

### Ocean Environments (`pufferlib/ocean/`)

54 built-in C game environments with Python bindings. Each environment has:
- `binding.c` — C source implementing the game logic
- Python module using the compiled binding
- Factory functions in `pufferlib/ocean/environment.py`

C bindings use a shared template (`env_binding.h`) and link against Raylib for rendering. Box2D is used by `impulse_wars`.

### Configuration

INI files in `pufferlib/config/` define per-environment hyperparameters (learning rate, batch size, network architecture, reward scaling, etc.). `default.ini` provides base values.

### Key Design Patterns

- **In-place buffer updates**: Environments write directly into pre-allocated numpy arrays (observations, rewards, terminals) rather than returning new arrays
- **Lazy imports**: Heavy dependencies (torch, specific env packages) imported only when needed
- **Structured dtype conversion**: `pufferlib/pytorch.py` handles converting complex numpy structured arrays to PyTorch tensors via `nativize_dtype()` / `nativize_tensor()`
- **Template C bindings**: New Ocean environments follow the `binding.c` pattern with `env_binding.h` macros

## Platform Notes

- macOS: Links Cocoa, OpenGL, IOKit frameworks
- Linux: Standard linking with `-Bsymbolic-functions`
- `setup.cfg` forces `inplace=true` and `force=true` for `build_ext`
- numpy<2.0 required throughout
