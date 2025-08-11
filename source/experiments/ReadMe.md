# 🧪 Running Experiments

This repository provides a structured framework for conducting reinforcement learning (RL) experiments with a focus on *
*Proximal Policy Optimization (PPO)** and its enhanced variant **IntrinsicAttentionPPO**. Experiments are orchestrated
using [Hydra](https://hydra.cc/) for configuration management and [SMAC](https://github.com/automl/SMAC3) for
hyperparameter optimization (HPO).

> **Note**: This section is research-focused and intended for reproducibility.

---

## ⚙️ Hydra Overview

We use **Hydra** to manage experimental configurations in a modular and composable way. This allows easy switching
between setups, reproducible experiments, and straightforward HPO integration.

**Key features used in this project:**

* **Config composition:** Combine base configs with specific overrides.
* **Multi-run mode (`-m`):** Automatically run sweeps for multiple configurations or seeds.
* **Structured config files:** Stored in `source/experiments/configs`.

For more information, see [Hydra Documentation](https://hydra.cc/docs/intro/).

---

# Experiments

## 🔍 Hyperparameter Optimization (HPO)

Before producing final results, a **hyperparameter search** was performed using **SMAC**. Search spaces are defined in
the configuration files found in:

```
source/experiments/configs
```

### PPO (Baseline)

* **HPO Config:** [`Umbrella_PPO_HPO.yaml`](../../source/experiments/configs/Umbrella_PPO_HPO.yaml)
* **Run File:** [`Umbrella_PPO_HPO.py`](../../source/experiments/src/Umbrella_PPO_HPO.py)
* **Results Directory:** [`experiment_data/hpo/ppo`](../../experiment_data/hpo/ppo)

**Re-run Command:**

```bash
python source/experiments/src/Umbrella_PPO_HPO.py -m
```

### IntrinsicAttentionPPO

* **HPO Config:** [`Umbrella_intrinsic_HPO.yaml`](../../source/experiments/configs/Umbrella_intrinsic_HPO.yaml)
* **Run File:** [`Umbrella_intrinsic_HPO.py`](../../source/experiments/src/Umbrella_intrinsic_HPO.py)
* **Results Directory:** [`experiment_data/hpo/intrinsic_attention`](../../experiment_data/hpo/intrinsic_attention)

**Re-run Command:**

```bash
python source/experiments/src/Umbrella_intrinsic_HPO.py -m
```

---

## 🎯 Environment Learning

After identifying optimal hyperparameters via HPO, the following main experiments were conducted.

### PPO

* **Config:** [`Umbrella_PPO_Experiment.yaml`](../../source/experiments/configs/Umbrella_PPO_Experiment.yaml)
* **Run File:** [`Umbrella_PPO_Experiment.py`](../../source/experiments/src/Umbrella_PPO_Experiment.py)
* **Results Directory:** [`experiment_data/UmbrellaPPO`](../../experiment_data/UmbrellaPPO)

**Re-run Command:**

```bash
python source/experiments/src/Umbrella_PPO_Experiment.py -m
```

### IntrinsicAttentionPPO

* **Config:** [
  `Umbrella_intrinsic_Experiment.yaml`](../../source/experiments/configs/Umbrella_intrinsic_Experiment.yaml)
* **Run File:** [`Umbrella_intrinsic_Experiment.py`](../../source/experiments/src/Umbrella_intrinsic_Experiment.py)
* **Results Directory:** [
  `experiment_data/UmbrellaIntrinsicAttentionPPO`](../../experiment_data/UmbrellaIntrinsicAttentionPPO)

**Re-run Command:**

```bash
python source/experiments/src/Umbrella_intrinsic_Experiment.py -m
```

---

## 📂 Directory Structure

```
source/
├── experiments/
│   ├── configs/      # Hydra configuration files
│   ├── src/          # Experiment run scripts
│   ├── plots/        # Plot Generation from experiment results
│   └── utils/        # Helper functions (if applicable)
experiment_data/
├── hpo/              # Results from hyperparameter optimization
├── UmbrellaPPO/      # Main PPO experiment results
└── UmbrellaIntrinsicAttentionPPO/ # Main IntrinsicAttentionPPO results
```

---

## 📜 Reproducibility Notes

* All experiments are seed-controlled via Hydra configs.
* Multi-run sweeps (`-m`) ensure parallel testing across parameter sets.
* SMAC configurations are explicitly stored for transparency.

---

## 📖 References

* [Hydra Documentation](https://hydra.cc/docs/intro/)
* [SMAC3 GitHub Repository](https://github.com/automl/SMAC3)
* [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)
* Intrinsic Attention mechanisms in RL (referenced in internal documentation)
