from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from model1_common import ensure_parent, load_json, load_trials


DEFAULT_CONFIG_PATH = Path("config/model1_track_b_item_stress_split.json")


@dataclass
class Config:
    base_trials_path: Path
    processed_trials_path: Path
    heldout_items_path: Path
    summary_path: Path
    heldout_item_fraction: float
    hash_salt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic Track B item-shift stress dataset."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config JSON.")
    return parser.parse_args()


def load_config(path: Path) -> Config:
    raw = load_json(path)
    cfg = Config(
        base_trials_path=Path(raw["base_trials_path"]),
        processed_trials_path=Path(raw["processed_trials_path"]),
        heldout_items_path=Path(raw["heldout_items_path"]),
        summary_path=Path(raw["summary_path"]),
        heldout_item_fraction=float(raw["heldout_item_fraction"]),
        hash_salt=str(raw.get("hash_salt", "track-b-item-stress")),
    )
    if not (0.0 < cfg.heldout_item_fraction < 1.0):
        raise ValueError("heldout_item_fraction must be between 0 and 1.")
    return cfg


def item_key(item_id: str, salt: str) -> str:
    payload = f"{salt}:{item_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def choose_heldout_items(train_items: list[str], cfg: Config) -> pd.DataFrame:
    items = pd.DataFrame({"item_id": pd.Series(sorted(train_items), dtype="string")})
    items["item_key"] = items["item_id"].map(lambda value: item_key(str(value), cfg.hash_salt))
    items = items.sort_values(["item_key", "item_id"], kind="mergesort").reset_index(drop=True)
    heldout_count = max(1, int(math.floor(len(items) * cfg.heldout_item_fraction)))
    items["heldout_for_item_shift"] = 0
    items.loc[: heldout_count - 1, "heldout_for_item_shift"] = 1
    items["heldout_for_item_shift"] = items["heldout_for_item_shift"].astype("int64")
    return items


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize_stress_dataset(
    track_b_item_stress: pd.DataFrame,
    heldout_items: pd.DataFrame,
    removed_train_rows: int,
    cfg: Config,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "config": {
            "base_trials_path": str(cfg.base_trials_path),
            "processed_trials_path": str(cfg.processed_trials_path),
            "heldout_items_path": str(cfg.heldout_items_path),
            "summary_path": str(cfg.summary_path),
            "heldout_item_fraction": cfg.heldout_item_fraction,
            "hash_salt": cfg.hash_salt,
        },
        "heldout_items": {
            "count": int(heldout_items["heldout_for_item_shift"].sum()),
            "share_of_original_train_items": float(heldout_items["heldout_for_item_shift"].mean()),
        },
        "rows": {
            "total_rows": int(len(track_b_item_stress)),
            "removed_train_rows": int(removed_train_rows),
            "train_rows": int((track_b_item_stress["split"] == "train").sum()),
            "validation_rows": int((track_b_item_stress["split"] == "validation").sum()),
            "test_rows": int((track_b_item_stress["split"] == "test").sum()),
        },
        "eval_item_overlap": {},
    }

    for split_name in ["validation", "test"]:
        row_mask = track_b_item_stress["split"] == split_name
        total_rows = int(row_mask.sum())
        unseen = int(track_b_item_stress.loc[row_mask, "new_item_in_test"].sum())
        seen = total_rows - unseen
        summary["eval_item_overlap"][split_name] = {
            "rows": total_rows,
            "seen_item_rows": seen,
            "new_item_rows": unseen,
            "new_item_share": (unseen / total_rows) if total_rows else 0.0,
        }

    return summary


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    track_b = load_trials(cfg.base_trials_path).copy()
    if not {"train", "validation", "test"}.issubset(set(track_b["split"].astype("string").unique().tolist())):
        raise ValueError("Base trials file must already contain Track B train/validation/test splits.")

    train_items = sorted(track_b.loc[track_b["split"] == "train", "item_id"].astype("string").unique().tolist())
    heldout_items = choose_heldout_items(train_items, cfg)
    heldout_item_ids = set(
        heldout_items.loc[heldout_items["heldout_for_item_shift"] == 1, "item_id"].astype("string").tolist()
    )

    track_b["heldout_item_for_stress"] = track_b["item_id"].isin(heldout_item_ids).astype("int64")
    train_holdout_mask = (track_b["split"] == "train") & (track_b["heldout_item_for_stress"] == 1)
    removed_train_rows = int(train_holdout_mask.sum())
    track_b_item_stress = track_b.loc[~train_holdout_mask].copy()

    remaining_train_items = set(
        track_b_item_stress.loc[track_b_item_stress["split"] == "train", "item_id"].astype("string").tolist()
    )
    eval_rows = track_b_item_stress["split"].isin(["validation", "test"])
    item_seen = track_b_item_stress["item_id"].isin(remaining_train_items)
    track_b_item_stress["item_seen_in_train"] = item_seen.astype("int64")
    track_b_item_stress["new_item_in_test"] = (eval_rows & ~item_seen).astype("int64")
    track_b_item_stress["primary_eval_eligible"] = (eval_rows & item_seen).astype("int64")

    ensure_parent(cfg.processed_trials_path)
    track_b_item_stress.to_csv(cfg.processed_trials_path, index=False)

    ensure_parent(cfg.heldout_items_path)
    heldout_items.to_csv(cfg.heldout_items_path, index=False)

    summary = summarize_stress_dataset(track_b_item_stress, heldout_items, removed_train_rows, cfg)
    write_json(cfg.summary_path, summary)

    print(f"Saved Track B item-shift stress trials to {cfg.processed_trials_path}")
    print(f"Saved heldout item list to {cfg.heldout_items_path}")
    print(f"Saved item-shift summary to {cfg.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
