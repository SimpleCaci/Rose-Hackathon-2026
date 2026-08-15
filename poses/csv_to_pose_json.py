"""
CSV -> Pose JSON builder

Assumptions (matches your dataset screenshot):
- First column is a label/path (e.g., "TRAIN/Downward_Dog/Images/downward_dog1.jpg")
- All other columns are numeric angle features (floats)

Output:
{
  "poses": [
    {
      "pose_id": "...",
      "name": "...",
      "hold_ms": 900,
      "points": 10,
      "target_angles": { "<col>": <mean> ... },
      "tolerances": { "<col>": <tol> ... },
      "template_landmarks": {}   # optional; left empty unless you add a source
    },
    ...
  ]
}
"""

import argparse
import json
import math
import os
import re
from typing import Dict, Any, List

import pandas as pd


def title_case_from_pose_id(pose_id: str) -> str:
    # "downward_dog" -> "Downward Dog"
    return re.sub(r"\s+", " ", pose_id.replace("_", " ").strip()).title()


def pose_id_from_label(label: str) -> str:
    """
    Robust-ish parsing for common label/path patterns.

    Examples:
      "TRAIN/Downward_Dog/Images/downward_dog1.jpg" -> "downward_dog"
      "Downward_Dog" -> "downward_dog"
      "chair_pose_12.jpg" -> "chair_pose"
    """
    if not isinstance(label, str):
        return "unknown_pose"

    parts = label.replace("\\", "/").split("/")

    # If it looks like TRAIN/<POSE_NAME>/..., use that folder.
    if len(parts) >= 2 and parts[0].upper() in {"TRAIN", "TEST", "VAL", "VALID", "VALIDATION"}:
        candidate = parts[1]
        return re.sub(r"[^a-z0-9_]+", "_", candidate.lower()).strip("_")

    # Otherwise try: filename without extension, stripping trailing digits
    base = os.path.splitext(parts[-1])[0].lower()
    base = re.sub(r"\d+$", "", base)  # downward_dog1 -> downward_dog
    base = re.sub(r"[^a-z0-9_]+", "_", base).strip("_")
    return base or "unknown_pose"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_pose_json(
    df: pd.DataFrame,
    label_col: str,
    hold_ms: int,
    points: int,
    tol_mode: str,
    tol_default: float,
    tol_sigma_mult: float,
    tol_min: float,
    tol_max: float,
    decimals: int,
    include_template_landmarks: bool,
) -> Dict[str, Any]:
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    if hold_ms <= 0:
        raise ValueError("hold_ms must be positive")
    if points <= 0:
        raise ValueError("points must be positive")

    # Ensure label is string
    df[label_col] = df[label_col].astype(str)

    # All non-label columns treated as numeric angles
    angle_cols = [c for c in df.columns if c != label_col]
    if not angle_cols:
        raise ValueError("At least one angle column is required")

    # Coerce numeric
    for c in angle_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    poses: List[Dict[str, Any]] = []

    for pose_id, g in df.groupby(df[label_col].map(pose_id_from_label)):
        # Means and stddevs per angle column
        means = g[angle_cols].mean(numeric_only=True)
        stds = g[angle_cols].std(numeric_only=True).fillna(0.0)

        target_angles = {
            col: round(float(means[col]), decimals)
            for col in angle_cols
            if not math.isnan(float(means[col]))
        }

        if tol_mode == "default":
            tolerances = {col: round(float(tol_default), decimals) for col in target_angles.keys()}
        else:  # "sigma"
            tolerances = {
                col: round(float(clamp(stds[col] * tol_sigma_mult, tol_min, tol_max)), decimals)
                for col in target_angles.keys()
            }

        pose_obj: Dict[str, Any] = {
            "pose_id": pose_id,
            "name": title_case_from_pose_id(pose_id),
            "hold_ms": hold_ms,
            "points": points,
            "target_angles": target_angles,
            "tolerances": tolerances,
        }

        if include_template_landmarks:
            # Your CSV does not contain landmarks, so this is a placeholder.
            # If you later add landmarks (e.g., separate JSON or columns), populate here.
            pose_obj["template_landmarks"] = {}

        poses.append(pose_obj)

    return {"poses": poses}


def main():
    ap = argparse.ArgumentParser(description="Convert pose-angle CSV into your JSON format.")
    ap.add_argument("--csv", required=True, help="Path to input CSV (e.g., Dataset_Downward_Dog_Angles.csv)")
    ap.add_argument("--out", required=True, help="Path to output JSON (e.g., poses.json)")
    ap.add_argument("--label-col", default=None, help="Label/path column name; default = first column")
    ap.add_argument("--hold-ms", type=int, default=900)
    ap.add_argument("--points", type=int, default=10)

    # Tolerance strategies:
    # 1) default: constant tolerance for every angle
    # 2) sigma: tolerance = stddev * multiplier, clamped to [min,max]
    ap.add_argument("--tol-mode", choices=["default", "sigma"], default="sigma")
    ap.add_argument("--tol-default", type=float, default=20.0)
    ap.add_argument("--tol-sigma-mult", type=float, default=2.0)
    ap.add_argument("--tol-min", type=float, default=8.0)
    ap.add_argument("--tol-max", type=float, default=35.0)

    ap.add_argument("--decimals", type=int, default=0, help="Rounding decimals for output numbers")
    ap.add_argument("--include-template-landmarks", action="store_true")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    label_col = args.label_col or df.columns[0]

    data = build_pose_json(
        df=df,
        label_col=label_col,
        hold_ms=args.hold_ms,
        points=args.points,
        tol_mode=args.tol_mode,
        tol_default=args.tol_default,
        tol_sigma_mult=args.tol_sigma_mult,
        tol_min=args.tol_min,
        tol_max=args.tol_max,
        decimals=args.decimals,
        include_template_landmarks=args.include_template_landmarks,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(data['poses'])} poses to: {args.out}")


if __name__ == "__main__":
    main()


#python csv_to_pose_json.py --csv Dataset_Downward_Dog_Angles.csv --out poses.json
