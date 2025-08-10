from __future__ import annotations

from typing import Any, Dict, List

import json
from pathlib import Path
from pprint import pprint


def find_data_dir(start: Path) -> Path:
    """
    Return the data directory assuming the program is started from the repo root.

    Notes:
        The `start` parameter is ignored. This function always resolves "./data"
        relative to the current working directory.

    Returns:
        Path: Absolute path to the "./data" directory.

    Raises:
        FileNotFoundError: If "./data" does not exist or is not a directory.
    """
    data_dir = Path.cwd() / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Erwarte Datenordner unter: {data_dir}. Starte das Programm aus dem Repo-Root."
        )
    return data_dir


def find_latest_ppo_run(data_dir: Path) -> Path:
    """
    Find the most recently modified PPO run directory under the given data dir.

    A PPO run directory is identified by its name starting with "PPO_".
    The latest directory is selected by modification time (stat().st_mtime).

    Args:
        data_dir (Path): Directory that contains PPO_* run folders.

    Returns:
        Path: Path to the latest PPO_* run directory.

    Raises:
        FileNotFoundError: If no PPO_* directories are found under data_dir.
    """
    ppo_runs = [
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("PPO_")
    ]
    if not ppo_runs:
        raise FileNotFoundError(f"No PPO_* runs found under {data_dir}")
    return max(ppo_runs, key=lambda p: p.stat().st_mtime)


def trial_dirs(run_dir: Path) -> List[Path]:
    """
    List trial directories within a PPO run directory.

    Trials are now located one level deeper in the directory hierarchy. This function
    searches each immediate subdirectory of run_dir for folders whose names start with "PPO_".

    Args:
        run_dir (Path): A PPO_* run directory that contains a subdirectory with trial folders.

    Returns:
        List[Path]: All trial directories matching "PPO_*" that are directories.
    """
    trials = []
    for subdir in run_dir.iterdir():
        if subdir.is_dir():
            trials.extend(
                [
                    d
                    for d in subdir.iterdir()
                    if d.is_dir() and d.name.startswith("PPO_")
                ]
            )
    return trials


def load_evaluation_records(result_json_path: Path) -> Dict[str, Any]:
    """
    Parse evaluation records from a Ray RLlib result.json file (newline-delimited JSON).

    For each line in result.json:
      - Extract the seed from rec["config"]["seed"].
      - Collect evaluation metrics from flat keys starting with "evaluation".
      - Add context fields if present: "training_iteration", "timesteps_total".
      - Skip malformed JSON lines and lines without evaluation fields.

    Args:
        result_json_path (Path): Path to the result.json file.

    Returns:
        Dict[str, Any]: Dictionary with:
            - "seed": The detected seed value (or None).
            - "evaluations": List of evaluation records per iteration.
    """
    eval_records: List[Dict[str, Any]] = []
    seed_value = None

    with result_json_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            seed_value = rec["config"]["seed"]

            # Only collect flat "evaluation/..." keys
            flat_eval = {
                k: v
                for k, v in rec.items()
                if isinstance(k, str) and k.startswith("evaluation")
            }

            # Add some useful iteration context if present
            if flat_eval:
                if "training_iteration" in rec:
                    flat_eval["training_iteration"] = rec["training_iteration"]
                elif "iteration" in rec:
                    flat_eval["training_iteration"] = rec["iteration"]
                if "timesteps_total" in rec:
                    flat_eval["timesteps_total"] = rec["timesteps_total"]
                eval_records.append(flat_eval)

    return {"seed": seed_value, "evaluations": eval_records}


def main() -> None:
    """
    Collect and pretty-print evaluation results for the latest PPO run across seeds.

    Workflow:
        1) Resolve the data directory as "./data" relative to the current working dir.
        2) Find the latest PPO_* run folder by modification time.
        3) For each trial (seed) within that run, parse its result.json and extract
           evaluation metrics over all iterations.
        4) Group results by detected seed and pprint the complete structure.

    Side Effects:
        Prints the latest run path and a dict keyed by seed containing:
            - "trial_dir": Path to the trial directory.
            - "num_evaluations": Number of evaluation records found.
            - "evaluations": The list of evaluation dicts per iteration.
    """
    # Start from this file location and locate data dir upwards
    start = Path(__file__).resolve().parent
    data_dir = find_data_dir(start)
    last_run = find_latest_ppo_run(data_dir)

    print(f"Latest PPO run: {last_run}")
    results_by_seed: Dict[str, Dict[str, Any]] = {}

    for tdir in trial_dirs(last_run):
        rj = tdir / "result.json"
        if not rj.exists():
            continue
        payload = load_evaluation_records(rj)
        seed = str(
            payload.get("seed") if payload.get("seed") is not None else tdir.name
        )
        results_by_seed[seed] = {
            "trial_dir": str(tdir),
            "num_evaluations": len(payload["evaluations"]),
            "evaluations": payload["evaluations"],
        }

    if not results_by_seed:
        print("No evaluation results found.")
        return

    # Pretty print grouped by seed
    print("\nAll evaluation results by seed:")
    pprint(results_by_seed)


if __name__ == "__main__":
    main()
