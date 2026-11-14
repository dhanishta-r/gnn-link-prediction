"""Merge all Milestone 2 result files into the final tables.

Reads every `results/m2/raw_*.csv` (one file per team member, all written by
`evaluate.append_results`) and writes

  results/m2/summary_results.csv   mean, std and number of seeds for every
                                   dataset / model / config / setting / split
  results/m2/results_table.md      the final table for the README

How the final table is built
  - "default" is the untuned configuration (config == "default").
  - "tuned" is, for each dataset, model and setting, the configuration with
    the best mean VALIDATION score in that setting (AUC for easy, MRR for
    heart). The test set is never used to choose.
  - Only configurations that have test rows can be chosen, and the reported
    numbers are their test results (mean +/- std over seeds).

Command line
    python src/aggregate_m2.py
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

METRICS = ["AUC", "AP", "MRR", "Hits@1", "Hits@3", "Hits@10", "Hits@100"]
KEYS = ["dataset", "model", "config", "setting", "split"]
SELECT_BY = {"easy": "AUC", "heart": "MRR"}
DATASET_ORDER = ["cora", "citeseer", "pubmed"]
DATASET_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def load_raw(folder: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(folder, "raw_*.csv")))
    if not paths:
        raise SystemExit(f"No raw_*.csv files in {folder}")
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"config": str, "notes": str, "member": str})
        df["file"] = os.path.basename(p)
        frames.append(df)
        print(f"  {os.path.basename(p):28s} {len(df):6d} rows")
    raw = pd.concat(frames, ignore_index=True)
    raw["dataset"] = raw["dataset"].str.lower()
    raw["config"] = raw["config"].fillna("default")

    dup = raw.duplicated(KEYS + ["seed"], keep=False)
    if dup.any():
        print(f"\nWARNING: {int(dup.sum())} rows repeat the same dataset/model/config/setting/split/seed.")
        print("The last one is kept. Check that no run was appended twice:")
        print(raw.loc[dup, KEYS + ["seed", "file"]].drop_duplicates().head(10).to_string(index=False))
        raw = raw.drop_duplicates(KEYS + ["seed"], keep="last")
    return raw


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(KEYS, sort=False)
    out = grouped[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.fillna(0.0)                              # one seed: std shown as 0
    out.insert(0, "n_seeds", grouped["seed"].nunique())
    return out.reset_index()


def choose_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """One 'default' and one 'tuned' row per dataset / model / setting."""
    rows = []
    test = summary[summary["split"] == "test"]
    valid = summary[summary["split"] == "valid"]
    for (dataset, model, setting), grp in test.groupby(["dataset", "model", "setting"], sort=False):
        default = grp[grp["config"] == "default"]
        if len(default):
            rows.append({**default.iloc[0].to_dict(), "version": "default", "valid_score": None})
        tuned = grp[grp["config"] != "default"]
        if not len(tuned):
            continue
        metric = SELECT_BY[setting]
        v = valid[(valid["dataset"] == dataset) & (valid["model"] == model) & (valid["setting"] == setting)]
        scores = v.set_index("config")[f"{metric}_mean"].to_dict()
        with_valid = [c for c in tuned["config"] if c in scores]
        if with_valid:
            best = max(with_valid, key=lambda c: scores[c])
            score = scores[best]
        else:
            best, score = tuned.sort_values("n_seeds", ascending=False)["config"].iloc[0], None
            print(f"WARNING: {dataset}/{model}/{setting}: no validation rows, cannot check how "
                  f"'{best}' was chosen.")
        rows.append({**tuned[tuned["config"] == best].iloc[0].to_dict(), "version": "tuned",
                     "valid_score": score})
    return pd.DataFrame(rows)


def _pm(row, metric) -> str:
    return f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_std']:.2f}"


def markdown_table(final: pd.DataFrame) -> str:
    lines = [
        "| Dataset | Model | Version | Easy AUC | Easy AP | HeaRT MRR | HeaRT Hits@10 | Seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    datasets = [d for d in DATASET_ORDER if d in set(final["dataset"])] + \
               sorted(set(final["dataset"]) - set(DATASET_ORDER))
    for dataset in datasets:
        d = final[final["dataset"] == dataset]
        for model in d["model"].drop_duplicates():
            for version in ("default", "tuned"):
                part = d[(d["model"] == model) & (d["version"] == version)]
                if not len(part):
                    continue
                easy = part[part["setting"] == "easy"]
                heart = part[part["setting"] == "heart"]
                e = easy.iloc[0] if len(easy) else None
                h = heart.iloc[0] if len(heart) else None
                seeds = int((e if e is not None else h)["n_seeds"])
                lines.append(
                    f"| {DATASET_NAMES.get(dataset, dataset)} | {model} | {version} | "
                    f"{_pm(e, 'AUC') if e is not None else '—'} | {_pm(e, 'AP') if e is not None else '—'} | "
                    f"{_pm(h, 'MRR') if h is not None else '—'} | {_pm(h, 'Hits@10') if h is not None else '—'} | "
                    f"{seeds} |")
    return "\n".join(lines)


def chosen_configs_table(final: pd.DataFrame) -> str:
    tuned = final[final["version"] == "tuned"]
    if not len(tuned):
        return "_No tuned configurations yet._"
    lines = ["| Dataset | Model | Setting | Chosen configuration | Validation score |", "|---|---|---|---|---|"]
    for _, r in tuned.iterrows():
        score = "—" if r["valid_score"] is None or pd.isna(r["valid_score"]) else \
            f"{r['valid_score']:.2f} ({SELECT_BY[r['setting']]})"
        lines.append(f"| {DATASET_NAMES.get(r['dataset'], r['dataset'])} | {r['model']} | {r['setting']} | "
                     f"`{r['config']}` | {score} |")
    return "\n".join(lines)


def main(folder: str = os.path.join("results", "m2")):
    print("Reading:")
    raw = load_raw(folder)
    summary = summarise(raw)
    summary.to_csv(os.path.join(folder, "summary_results.csv"), index=False, float_format="%.4f")
    final = choose_configs(summary)

    text = "\n".join([
        "<!-- Generated by src/aggregate_m2.py. Do not edit by hand. -->",
        "",
        "Test results, mean ± standard deviation over seeds, in %. "
        "Easy = random negatives; HeaRT = 500 hard negatives per positive.",
        "",
        markdown_table(final),
        "",
        "Tuned configurations, chosen on the validation set only "
        "(AUC in the easy setting, MRR in the HeaRT setting):",
        "",
        chosen_configs_table(final),
        "",
    ])
    with open(os.path.join(folder, "results_table.md"), "w") as f:
        f.write(text)
    print(f"\nWrote {folder}/summary_results.csv and {folder}/results_table.md\n")
    print(text)
    return summary, final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=os.path.join("results", "m2"))
    main(parser.parse_args().folder)
