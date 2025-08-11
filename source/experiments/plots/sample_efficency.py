from typing import Dict, List, Union

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils

PPO_DATA = "./experiment_data/UmbrellaPPO"
INTRINSICATTENTION_DATA = "./experiment_data/UmbrellaIntrinsicAttentionPPO"


## Inspiration from https://colab.research.google.com/drive/1a0pSD-1tWhMmeJeeoyZM1A-HCW3yf1xR?usp=sharing#scrollTo=-xahXi1brHuf ##


def collect_result_entries(data_root: Union[str, Path]) -> List[Dict[str, object]]:
    """
    Recursively searches the specified data directory for 'result.json' files and extracts metadata for each file.

    Returns a list of dictionaries, each containing:
        - 'result_json': Path to the result file.
        - 'seed': Seed value extracted from the directory name.
        - 'length': Environment length extracted from the directory name.

    Only files matching the expected directory pattern are included.
    """
    root = Path(data_root)
    entries: List[Dict[str, object]] = []
    pattern = re.compile(r"seed(\d+).*length(\d+)", re.IGNORECASE)

    for rj in root.rglob("result.json"):
        two_up_name = rj.parent.parent.name
        m = pattern.search(two_up_name)
        if not m:
            print("Pattern passt nicht")
            continue
        seed = int(m.group(1))
        length = int(m.group(2))
        entries.append(
            {
                "result_json": str(rj),
                "seed": seed,
                "length": length,
            }
        )
    return entries


def extract_episode_returns(
    entries: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Loads the list of episode return means from each result.json file under
    'evaluation/env_runners/episode_return_mean'.

    Returns a list of dictionaries, each containing:
        - 'length': Environment length.
        - 'seed': Seed value.
        - 'episode_return_mean': List of episode return means for that seed and length.
    """
    results = []
    for entry in entries:
        path = entry["result_json"]
        episode_returns = []
        with open(path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    erm = (
                        data.get("evaluation", {})
                        .get("env_runners", {})
                        .get("episode_return_mean")
                    )
                    if erm is not None:
                        episode_returns.append(erm)
                    else:
                        raise NotImplementedError
                except Exception:
                    raise NotImplementedError

        results.append(
            {
                "length": entry["length"],
                "seed": entry["seed"],
                "episode_return_mean": episode_returns,
            }
        )
    return results


def aggregate_by_length(results: List[Dict[str, object]]) -> Dict[int, np.ndarray]:
    """
    Aggregates episode return means by environment length.

    Returns a dictionary:
        key: length
        value: numpy array of shape (n_seeds, 1, n_evals), padded with np.nan if necessary.
    """
    length_dict: Dict[int, List[List[float]]] = {}
    for entry in results:
        length = entry["length"]
        episode_returns = entry["episode_return_mean"]
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(episode_returns)
    out = {}
    for length, returns_list in length_dict.items():
        max_len = max(len(l) for l in returns_list)
        padded = [l + [np.nan] * (max_len - len(l)) for l in returns_list]
        arr = np.array(padded, dtype=float)  # Shape (n_seeds, n_evals)
        arr = arr[:, np.newaxis, :]  # Shape (n_seeds, 1, n_evals)
        out[length] = arr
    return out


def plot_sample_efficiency_comparison(
    ppo_data: Dict[int, np.ndarray],
    intrinsic_data: Dict[int, np.ndarray],
    save_dir="rliable_plots",
):
    """
    Creates sample efficiency plots for PPO and IntrinsicAttentionPPO for each environment length.

    Plots IQM and confidence intervals over training steps and saves each plot as a PNG file.
    """
    Path(save_dir).mkdir(exist_ok=True)
    # Do zeropadding for missing values
    for length, arr in ppo_data.items():
        ppo_data[length] = np.nan_to_num(arr, nan=0.0)
    for length, arr in intrinsic_data.items():
        intrinsic_data[length] = np.nan_to_num(arr, nan=0.0)
    for length in ppo_data.keys():
        ppo_scores = ppo_data[length]
        intrinsic_scores = intrinsic_data[length]

        data = {
            "PPO": ppo_scores,
            "IntrinsicAttentionPPO": intrinsic_scores,
        }
        # Hardcoded Training Step Size
        training_steps = np.arange(ppo_scores.shape[2]) * 5000

        def iqm(scores):
            """
            Compute the Inverse of the Interquartile Mean (IQM) for given scores.
            """
            return np.array(
                [
                    metrics.aggregate_iqm(scores[..., frame])
                    for frame in range(scores.shape[-1])
                ]
            )

        iqm_scores, iqm_cis = rly.get_interval_estimates(data, iqm, reps=10000)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_utils.plot_sample_efficiency_curve(
            training_steps / 1000,
            iqm_scores,
            iqm_cis,
            algorithms=list(data.keys()),
            xlabel="Training Steps (thousands)",
            ylabel="Evaluation Return",
            ax=ax,
        )
        ax.set_title(
            f"Sample Efficiency (IQM ± CI): PPO vs IntrinsicAttentionPPO (length={length}) across 10 Seeds"
        )
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_efficiency_length_{length}.png", dpi=300)
        plt.close(fig)


def plot_sample_efficiency_final_vs_length(
    ppo_data: Dict[int, np.ndarray],
    intrinsic_data: Dict[int, np.ndarray],
    save_path="rliable_plots/sample_efficiency_final_vs_length.png",
):
    """
    Creates a summary plot with environment length on the x-axis and IQM of the final evaluation return (with uncertainty) on the y-axis.

    Plots IQM and confidence intervals for PPO and IntrinsicAttentionPPO across all lengths and saves the plot as a PNG file.
    """
    # Do zeropadding for missing values
    for length, arr in ppo_data.items():
        ppo_data[length] = np.nan_to_num(arr, nan=0.0)
    for length, arr in intrinsic_data.items():
        intrinsic_data[length] = np.nan_to_num(arr, nan=0.0)

    lengths = sorted(ppo_data.keys())

    def iqm_last(scores):
        # scores: (n_seeds, 1, n_evals)
        return metrics.aggregate_iqm(scores[:, 0, -1])

    # Dict[str, np.ndarray] mit shape (n_seeds, 1, n_evals)
    data = {
        "PPO": np.stack(
            [ppo_data[length] for length in lengths], axis=1
        ),  # shape (n_seeds, len(lengths), 1, n_evals)
        "IntrinsicAttentionPPO": np.stack(
            [intrinsic_data[length] for length in lengths], axis=1
        ),
    }
    for alg in data:
        data[alg] = np.stack(
            [data[alg][:, i, 0, -1] for i in range(len(lengths))], axis=1
        )

    def iqm_curve(scores):
        return np.array(
            [metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])]
        )

    iqm_scores, iqm_cis = rly.get_interval_estimates(data, iqm_curve, reps=10000)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_utils.plot_sample_efficiency_curve(
        lengths,
        iqm_scores,
        iqm_cis,
        algorithms=list(data.keys()),
        xlabel="Length",
        ylabel="Evaluation Return",
        ax=ax,
    )
    ax.set_title("Final Evaluation vs Length (IQM ± CI) across 10 Seeds")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # Fetch the Data and create the plots
    ppo_entries = collect_result_entries(PPO_DATA)
    intrinsic_entries = collect_result_entries(INTRINSICATTENTION_DATA)

    ppo_episode_returns = extract_episode_returns(ppo_entries)
    intrinsic_episode_returns = extract_episode_returns(intrinsic_entries)

    ppo_data = aggregate_by_length(ppo_episode_returns)
    intrinsic_data = aggregate_by_length(intrinsic_episode_returns)

    plot_sample_efficiency_comparison(ppo_data, intrinsic_data)

    print(len(ppo_entries), "PPO entries found")
    print(len(intrinsic_entries), "Intrinsic Attention entries found")
