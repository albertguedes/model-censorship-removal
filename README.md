# Model Censorship Removal (MCR)

Automatic neutralization of safety alignment constraints in transformer-based language models.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

MCR removes content policy enforcement ("safety alignment") from transformer-based language models without requiring post-training or fine-tuning. It combines directional ablation (Arditi et al. 2024, Lai 2025) with TPE-based hyperparameter optimization (Optuna) to find abliteration parameters that minimize refusal rate while preserving the model's original capabilities.

The process is fully automatic and requires no understanding of transformer internals. MCR accepts any causal language model from the HuggingFace model hub and produces a policy-neutral variant that retains the original model's intelligence and knowledge.

### What MCR does

- **Directional ablation**: Orthogonalizes attention output and MLP down-projection matrices against the refusal direction vector
- **Multi-component optimization**: Treats attention and MLP interventions as separate optimization problems
- **Interpolated residual directions**: Explores the full continuous space between per-layer residual directions via float-index interpolation
- **Flexible ablation kernel**: Non-constant ablation weights over layers, shaped by learnable max/min weight and position parameters

### Supported architectures

- Dense transformers (Llama, Gemma, Qwen, Mistral, GPT-2, etc.)
- Multimodal models with text backbones
- MoE architectures (Mixtral, Qwen3.5-MoE, etc.)
- Hybrid models (Qwen3.5)

Unsupported: pure state-space models (Mamba, etc.), novel research architectures.

---

## Installation

```sh
git clone https://github.com/albertguedes/model-censorship-removal.git
cd model-censorship-removal
```

Requires Python 3.10+, PyTorch 2.2+ (hardware-specific version required). MCR uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
uv sync
```

---

## Quick Start

```sh
uv run mcr Qwen/Qwen3-4B-Instruct-2507
```

MCR will benchmark your hardware, load the model, compute residual directions from reference prompt sets, run the Optuna optimization study (200 trials by default), and present the Pareto-optimal results. You may then export the model, upload it to HuggingFace, run built-in benchmarks, or chat with the decensored variant.

For quantized inference (VRAM-constrained environments):

```sh
uv run mcr <model> --quantization bnb_4bit
```

---

## Benchmark Results

MCR achieves lower KL divergence from the original model than manual abliterations at equivalent refusal suppression:

| Model | Refusals (100 refusal-eliciting prompts) | KL divergence (100 neutral prompts) |
|:------|----------------------------------------:|-----------------------------------:|
| google/gemma-3-12b-it (original) | 97/100 | 0 |
| mlabonne/gemma-3-12b-it-abliterated-v2 | 3/100 | 1.04 |
| huihui-ai/gemma-3-12b-it-abliterated | 3/100 | 0.45 |
| p-e-w/gemma-3-12b-it-heretic | 3/100 | 0.16 |

*Values are platform- and hardware-dependent. Tested on PyTorch 2.8, RTX 5090.*

The lower KL divergence indicates MCR preserves more of the original model's capabilities while achieving the same refusal suppression. Independent community benchmarks on MMLU and GSM8K confirm favorable comparison with competing tools.

---

## Configuration

Default configuration covers most use cases. For advanced control:

```sh
mcr --help              # All CLI options
cat config.default.toml # All configuration parameters
```

Key parameters:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `n_trials` | 200 | Optimization trials |
| `n_startup_trials` | 60 | Random sampling trials before Bayesian optimization |
| `orthogonalize_direction` | true | Subtract only the component orthogonal to the refusal direction |
| `row_normalization` | "full" | LoRA adapter row normalization strategy |
| `quantization` | none | Model quantization (`bnb_4bit`, `bnb_8bit`) |

---

## Architecture

MCR implements a parametrized variant of directional ablation. For each supported transformer component (attention out-projection, MLP down-projection), it:

1. Computes residual directions as difference-of-means between refusal-eliciting and neutral prompt first-token residuals, per layer
2. Orthogonalizes the component's weight matrix against the residual direction, inhibiting that direction's expression
3. Wraps the orthogonalized matrix in a LoRA adapter for efficient storage
4. Optimizes ablation parameters via TPE to co-minimize refusal rate and KL divergence

### Key innovations

- **Continuous residual direction indexing**: Float indices interpolate between per-layer directions, exploring a vast direction space beyond integer indices
- **Per-component parameter optimization**: Attention and MLP interventions use separate ablation weight kernels, since MLP interventions tend to be more damaging
- **Flexible kernel shape**: Learnable max/min weight and position parameters adapt the ablation profile to each model's architecture

---

## Research Tools

MCR includes interpretability features for studying model internals. Install with the `research` extra:

```sh
uv sync --extra research
```

### Residual vector visualization (`--plot-residuals`)

Computes residual vectors for each transformer layer, projects them to 2D via PaCMAP, and generates layer-wise scatter plots and an animated GIF showing residual transformation across layers.

### Residual geometry analysis (`--print-residual-geometry`)

Produces a quantitative table of residual vector metrics per layer: inter-cluster cosine similarity, cluster norms, silhouette coefficients.

---

## Reproduction and Verification

MCR supports bit-exact reproducibility of published models. When uploading, MCR can generate a `reproduce.json` containing the full environment snapshot (package versions, PyTorch build, config, ablation parameters) and a SHA256SUMS file of model weights.

To reproduce a published MCR model:

```sh
mcr --reproduce reproduce.json
```

Place the included Optuna study journal in `checkpoints/` to export other Pareto-optimal trials from the same study.

---

## About

MCR is a fork of [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann, restructured for professional and enterprise use. The original Heretic project remains available for the community.

---

## Citation

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Fully automatic censorship removal for language models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```

---

## Acknowledgments

MCR builds on the directional ablation framework introduced by Arditi et al. (2024), with innovations motivated by Maxime Labonne's practical abliteration work and Jim Lai's projected and norm-preserving biprojected abliteration techniques.

---

## License

Copyright 2025-2026 Albert Mendes + Philipp Emanuel Weidmann

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See the GNU Affero General Public License for details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

**By contributing to this project, you agree to release your contributions under the same license.**
