from typing import Dict, List, Union

import re
from pathlib import Path

import numpy as np

PPO_DATA = "./experiment_data/UmbrellaPPO"
INTRINSICATTENTION_DATA = "./experiment_data/UmbrellaIntrinsicAttentionPPO"


## Inspiration from https://colab.research.google.com/drive/1a0pSD-1tWhMmeJeeoyZM1A-HCW3yf1xR?usp=sharing#scrollTo=-xahXi1brHuf ##


def collect_result_entries(data_root: Union[str, Path]) -> List[Dict[str, object]]:
    """
    Durchsuche den gegebenen Datenordner rekursiv nach 'result.json' und liefere
    eine Liste von Dicts mit:
      - result_json: Pfad zur jeweiligen 'result.json'
      - seed: Seed (aus dem Ordnernamen zwei Ebenen über der Datei)
      - length: Länge des Environments (aus dem Ordnernamen zwei Ebenen über der Datei)
    Einfach gehalten, ohne umfangreiches Fehlerhandling.
    """
    root = Path(data_root)
    entries: List[Dict[str, object]] = []
    pattern = re.compile(r"seed(\d+).*length(\d+)", re.IGNORECASE)

    for rj in root.rglob("result.json"):
        # Zwei Ebenen über der Datei: .../<seed_length_dir>/<trial_dir>/result.json
        two_up_name = rj.parent.parent.name
        m = pattern.search(two_up_name)
        if not m:
            print("Pattern passt nicht")
            continue  # schlicht überspringen, wenn nicht passend
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


import json


def extract_episode_returns(
    entries: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Extrahiere aus jeder result.json die Liste der episode_return_mean Werte unter
    evaluation/env_runners/episode_return_mean.
    Gibt eine Liste von Dicts zurück mit:
      - length
      - seed
      - episode_return_mean: Liste der Werte aus allen result.json-Dateien
    """
    results = []
    for entry in entries:
        path = entry["result_json"]
        episode_returns = []
        with open(path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Extrahiere episode_return_mean, falls vorhanden
                    erm = (
                        #  data.get("evaluation", {})
                        data.get("env_runners", {}).get("episode_return_mean")
                    )
                    if erm is not None:
                        episode_returns.append(erm)
                    else:
                        raise NotImplementedError
                except Exception:
                    raise NotImplementedError
                    continue  # einfach überspringen

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
    Aggregiert die Ausgabe von extract_episode_returns nach length.
    Gibt ein Dictionary zurück:
      key: length
      value: numpy array mit Shape (n_seeds, 1, n_evals)
    Padding mit np.nan, falls die Listen unterschiedlich lang sind.
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


import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics, plot_utils


def plot_sample_efficiency_comparison(
    ppo_data: Dict[int, np.ndarray],
    intrinsic_data: Dict[int, np.ndarray],
    save_dir="rliable_plots",
):
    """
    Vergleicht PPO und IntrinsicAttentionPPO für jede Länge in einem Sample Efficiency Plot.
    Speichert die Plots als PNG im angegebenen Verzeichnis.
    """
    Path(save_dir).mkdir(exist_ok=True)
    for length in ppo_data.keys():
        print(f"Länge {length}: PPO-Daten (erste Zeile): {ppo_data[length][0, 0, :]}")
        print(
            f"Länge {length}: Intrinsic-Daten (erste Zeile): {intrinsic_data[length][0, 0, :]}"
        )
    for length, arr in ppo_data.items():
        ppo_data[length] = np.nan_to_num(arr, nan=0.0)
    for length, arr in intrinsic_data.items():
        intrinsic_data[length] = np.nan_to_num(arr, nan=0.0)
    for length in ppo_data.keys():
        # Daten vorbereiten: Shape (n_seeds, 1, n_evals) -> (n_seeds, n_evals)

        ppo_scores = ppo_data[length]
        intrinsic_scores = intrinsic_data[length]

        # RLiable erwartet Dict[str, np.ndarray]
        data = {
            "PPO": ppo_scores,
            "IntrinsicAttentionPPO": intrinsic_scores,
        }
        # Trainingsschritte: z.B. alle 5000 Schritte evaluiert
        training_steps = np.arange(ppo_scores.shape[2]) * 5000  # Passe 5000 ggf. an!

        def iqm(scores):
            """
            Compute the Inverse of the Interquartile Mean (IQM) for given scores.
            """
            return np.array(
                [
                    metrics.aggregate_mean(scores[..., frame])
                    for frame in range(scores.shape[-1])
                ]
            )

        iqm_scores, iqm_cis = rly.get_interval_estimates(data, iqm, reps=10000)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_utils.plot_sample_efficiency_curve(
            training_steps / 1000,  # in Tausend Schritten
            iqm_scores,
            iqm_cis,
            algorithms=list(data.keys()),
            xlabel="Training Steps (thousands)",
            ylabel="Mean Evaluation Return",
            ax=ax,
        )
        ax.set_title(
            f"Sample Efficiency: PPO vs IntrinsicAttentionPPO (length={length})"
        )
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_efficiency_length_{length}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    ppo_entries = collect_result_entries(PPO_DATA)
    intrinsic_entries = collect_result_entries(INTRINSICATTENTION_DATA)

    ppo_episode_returns = extract_episode_returns(ppo_entries)
    intrinsic_episode_returns = extract_episode_returns(intrinsic_entries)

    ppo_data = aggregate_by_length(ppo_episode_returns)
    intrinsic_data = aggregate_by_length(intrinsic_episode_returns)

    plot_sample_efficiency_comparison(ppo_data, intrinsic_data)

    import pprint

    # pprint.pprint(ppo_episode_returns)
    # pprint.pprint(intrinsic_episode_returns)

    pprint.pprint(ppo_data)
    pprint.pprint(intrinsic_data)

    print(len(ppo_entries), "PPO entries found")
    print(len(intrinsic_entries), "Intrinsic Attention entries found")
