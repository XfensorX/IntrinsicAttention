# 🧠 Intrinsic Rewards with Attention Network on PPO learned by Meta-Gradient

> A research codebase exploring **meta-learning of intrinsic rewards** for PPO using an **attention-based intrinsic
reward
network** trained on **full episodes**. The policy optimizes for the sum of extrinsic and intrinsic rewards during PPO
> updates, while the intrinsic network itself is updated by a **meta-gradient** signal to improve **extrinsic return**.


![intrinsic_attention_umbrella_bnner.jpg](images/intrinsic_attention_umbrella_bnner.jpg)
---

<p align="center">
  <a href="https://pypi.org/project/numpy/">
    <img src="https://img.shields.io/pypi/v/numpy.svg?label=numpy&logo=numpy" alt="numpy"/>
  </a>
  <a href="https://pypi.org/project/torch/">
    <img src="https://img.shields.io/pypi/v/torch.svg?label=PyTorch&logo=pytorch" alt="PyTorch"/>
  </a>
  <a href="https://pypi.org/project/hydra-core/">
    <img src="https://img.shields.io/pypi/v/hydra-core.svg?label=hydra-core" alt="hydra-core"/>
  </a>
  <a href="https://pypi.org/project/ray/">
    <img src="https://img.shields.io/pypi/v/rllib.svg?label=rllib&logo=ray" alt="rllib"/>
  </a>
  <a href="https://pypi.org/project/matplotlib/">
    <img src="https://img.shields.io/pypi/v/matplotlib.svg?label=matplotlib" alt="matplotlib"/>
  </a>
  <a href="https://pypi.org/project/rliable/">
    <img src="https://img.shields.io/pypi/v/rliable.svg?label=rliable" alt="rliable"/>
  </a>
  <a href="https://pypi.org/project/pytest/">
    <img src="https://img.shields.io/pypi/v/pytest.svg?label=pytest&logo=pytest" alt="pytest"/>
  </a>
  <a href="https://pypi.org/project/jupyterlab/">
    <img src="https://img.shields.io/pypi/v/jupyter.svg?label=jupyterlab&logo=jupyter" alt="jupyterlab"/>
  </a>
</p>

## Overview

At a high level:

- A trajectory-level **Intrinsic Reward Network (IRN)** maps episode context (and optionally per-step features) to
  intrinsic rewards.
- PPO trains the policy on **extrinsic + intrinsic** rewards.
- A **meta-gradient step** adjusts the IRN parameters so that policy improvements increase
  **extrinsic** return.
- A dedicated **extrinsic value network** maintains PPO’s stability and credit assignment for the environment reward.

---

## Disclaimer on AI Assistance

Portions of this codebase and documentation were created **with the assistance of AI/LLMs**. Outputs were reviewed and
integrated by the authors.

---

## 🧩 Method Summary

We adapt PPO by:

1. **Intrinsic Reward Network:** Produces per-step intrinsic rewards from information aggregated across **full training
   episodes** via attention.
2. **PPO Training:** The policy is optimized on the **PPO loss** using the sum of **extrinsic + intrinsic** rewards.
3. **Meta-Gradient Update:** The intrinsic network is updated using the **meta-gradient** of the **policy’s extrinsic
   return**, ensuring intrinsic shaping serves the environment objective.
4. **Auxiliary Value Head:** A dedicated value network for the **extrinsic** reward is maintained for PPO.

---

## ⚙️ Quickstart (uv + .venv)

> Prerequisites: `python` (version 3.11.12), [`uv`](https://github.com/astral-sh/uv) installed.

```bash
# 1) Clone
git clone https://github.com/XfensorX/IntrinsicAttention.git
cd https://github.com/XfensorX/IntrinsicAttention

# 2) Create a local virtual environment
uv venv .venv

# 3) Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 4) Install dependencies from pyproject.toml
uv sync

```

### ✅ Running Tests

Running the tests which are located in [tests](tests)

```shell
uv run pytest
```

---

# 🧪 Running Experiments

<p align="center">
  <a href="./source/experiments/ReadMe.md">
    <img src="https://img.shields.io/badge/View-Experiment_Docs-green?style=for-the-badge&logoColor=white" alt="Experiment Docs Badge" />
  </a>
</p>

---

# 🏗️ Architecture & Algorithm

<p align="center">
  <a href="./source/intrinsic_attention_ppo/ReadMe.md">
    <img src="https://img.shields.io/badge/View-Algorithm_Docs-blue?style=for-the-badge&logoColor=white" alt="Algorithm Docs Badge" />
  </a>
</p>

---

# 📁 Project Structure

```markdown
.
├─ experiment_data/ # Raw and processed experiment outputs, logs, checkpoints
├─ images/ # Figures used in the README and paper (e.g., banner.png)
├─ plots/ # Notebooks and code to generate plots from experiment_data
├─ sandbox/ # Exploratory prototypes & research scratch space
├─ tests/ # Unit tests
└─ source/ # Project code

source:
├─ environments/ # Environment wrappers, vectorized envs, episode utils
├─ experiments/ # Configurations and scripts for launching experiments
└─ intrinsic_attention_ppo/ # Algorithm implementation

```

---

# 📊 Results

For results and further research information please have a look at our poster and paper:

- Poster: [poster.pdf](poster.pdf)
- Paper: [paper.pdf](paper.pdf)

---

# ℹ️ Further Information:

## 🔁 Reproducibility

- seed handling and configuration parameters are registered with hydra: [configs](source/experiments/configs)
- exact package versions can be found in [pyproject.toml](pyproject.toml)

## 📚 Related Work

Inspiration for this work was mainly from:

- **LIRPG** – Learning Intrinsic Rewards for Policy Gradient
    - 📄 [ArXiv: 1804.06459](https://arxiv.org/abs/1804.06459)
    - 💻 [GitHub: lirpg](https://github.com/Hwhitetooth/lirpg)

- Self-Attention based Temporal Intrinsic Reward for Reinforcement Learning
    - [IEEE Explore](https://ieeexplore.ieee.org/document/9727314)
    - ❌ No public code repository found.

- **Memory-RL** – Memory-based Reinforcement Learning
    - 📄 [ArXiv: 2307.03864](https://arxiv.org/abs/2307.03864)
    - 💻 [GitHub: Memory-RL](https://github.com/twni2016/Memory-RL)

## 📝 Cite This Work

@misc{intrinsic_attention_ppo,
title = {Intrinsic Rewards with Attention Network on PPO learned with Meta-Gradient},
author = {Julius Heidmann, Philipp Link, Luan Liebig-Schultz},
year = {2025},
note = {Code: https://github.com/XfensorX/IntrinsicAttention}
}

---

# 📚 References

1. Jiang, Z., Tian, D., Yang, Q., & Peng, Z. (2021). *Self-Attention based Temporal Intrinsic Reward for Reinforcement
   Learning*. In **2021 China Automation Congress (CAC)** (pp. 2022–2026).
   IEEE. [https://doi.org/10.1109/CAC53003.2021.9727314](https://doi.org/10.1109/CAC53003.2021.9727314)

2. Wu, Z., Liang, E., Luo, M., Mika, S., Gonzalez, J. E., & Stoica, I. (2021). *RLlib Flow: Distributed Reinforcement
   Learning is a Dataflow Problem*. In **Conference on Neural Information Processing Systems (NeurIPS)
   **. [PDF](https://proceedings.neurips.cc/paper/2021/file/2bce32ed409f5ebcee2a7b417ad9beed-Paper.pdf)

3. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). *Deep Reinforcement Learning
   at the Edge of the Statistical Precipice*. In **Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
   ** (pp.
   29304–29320). [Abstract](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

4. Ross, O. D. I., & L’Hermitte, J. (2020). *Hydra: A framework for elegantly configuring complex applications*. *
   *Journal of Open Source Software, 5**(52),
   2415. [https://doi.org/10.21105/joss.02415](https://doi.org/10.21105/joss.02415)

5. Osband, I., Doron, Y., Hessel, M., Aslanides, J., Sezener, E., Saraiva, A., McKinney, K., Lattimore, T., Szepesvári,
   C., Singh, S., Van Roy, B., Sutton, R. S., Silver, D., & van Hasselt, H. (2020). *Behaviour Suite for Reinforcement
   Learning*. In **8th International Conference on Learning Representations (ICLR 2020)
   **. [OpenReview](https://openreview.net/forum?id=rygf-kSYwH)

6. Ni, T., Ma, M., Eysenbach, B., & Bacon, P.-L. (2023). *When Do Transformers Shine in RL? Decoupling Memory from
   Credit Assignment*. In **Advances in Neural Information Processing Systems 36 (NeurIPS 2023)
   **. [Abstract](http://papers.nips.cc/paper_files/paper/2023/hash/9dc5accb1e4f4a9798eae145f2e4869b-Abstract-Conference.html)

7. Pignatelli, E., Ferret, J., Geist, M., Mesnard, T., van Hasselt, H., & Toni, L. (2024). *A Survey of Temporal Credit
   Assignment in Deep Reinforcement Learning*. **Transactions on Machine Learning Research, 2024
   **. [OpenReview](https://openreview.net/forum?id=bNtr6SLgZf)

8. Zheng, Z., Oh, J., & Singh, S. (2018). *On Learning Intrinsic Rewards for Policy Gradient Methods*. In **Proceedings
   of the 32nd International Conference on Neural Information Processing Systems (NIPS’18)** (pp. 4649–4659). Curran
   Associates, Inc.

9. Lindauer, M., Eggensperger, K., Feurer, M., Biedenkapp, A., Deng, D., Benjamins, C., Ruhkopf, T., Sass, R., & Hutter,
   F. (2022). *SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization*. **Journal of Machine
   Learning Research, 23**(54), 1–9. [http://jmlr.org/papers/v23/21-0888.html](http://jmlr.org/papers/v23/21-0888.html)

---

# Important Documentation Linked

# Docs RLlib

- [Config](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.html)
- [Evaluation](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html)
- [Training](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training.html)
- [Customizing of Model](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.core.rl_module.default_model_config.DefaultModelConfig.html)

# Implementation Examples (for new API-Stack)

- [Custom Loss Function](https://github.com/ray-project/ray/blob/master/rllib/examples/learners/classes/custom_ppo_loss_fn_learner.py)
- [Custom IntrinsicReward Learner](https://github.com/ray-project/ray/blob/master/rllib/examples/learners/classes/intrinsic_curiosity_learners.py)
- [Custom IntrinsicReward RLModule](https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/intrinsic_curiosity_model_rlm.py)
- [Recurrent RLModule (LSTM) with InnerState](https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/lstm_containing_rlm.py)
- [Connectors(Custom TrainingBatches with PrevActPrevRew)](https://github.com/ray-project/ray/blob/master/rllib/examples/connectors/prev_actions_prev_rewards.py)
- [Intrinsic Reward Exmple](https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/intrinsic_curiosity_model_rlm.py)

