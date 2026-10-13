from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from kalm.keypoint_predictor import DINOFeatureExtractor
from kalm.keypoint_proposal import ProposalConfig, propose_keypoints


def load_mask_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    out = []
    for r in records:
        item = dict(r)
        item["mask"] = np.load(r["mask_file"]).astype(bool)
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Initial hybrid 2D KALM keypoint proposal")
    p.add_argument("--image", required=True)
    p.add_argument("--mask-json", required=True, help="masks.json")
    p.add_argument("--placement-json", default=None)
    p.add_argument("--output", default="output/kimi_keypoints")
    p.add_argument("--n-target-points", type=int, default=6)
    p.add_argument("--n-obstacle-points", type=int, default=6)
    p.add_argument("--n-placement-points", type=int, default=5)
    p.add_argument("--no-dino", action="store_true")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    masks = load_mask_json(args.mask_json)

    placement_mask = None
    if args.placement_json:
        with open(args.placement_json, "r", encoding="utf-8") as f:
            placement = json.load(f)
        if placement.get("mask_file"):
            placement_mask = np.load(placement["mask_file"]).astype(bool)

    dino = None
    if not args.no_dino:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading DINO on {device}...")
        extractor = DINOFeatureExtractor(device=device, input_size=448, output_size=256)
        dino = extractor.extract_feature_from_image(image)

    result = propose_keypoints(
        image=image,
        masks=masks,
        placement_mask=placement_mask,
        dino_feature_map=dino,
        config=ProposalConfig(
            n_object_boundary_points=args.n_target_points,
            n_obstacle_boundary_points=args.n_obstacle_points,
            n_placement_candidates=args.n_placement_points,
        ),
    )

    with (out / "keypoints_2d.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    for q in result["grasp_points"]:
        x, y = q["pixel_xy"]; plt.scatter([x], [y], marker="*", s=170, label="grasp")
    q = result["target_boundary_points"]
    if q:
        plt.scatter([p["pixel_xy"][0] for p in q], [p["pixel_xy"][1] for p in q], marker="o", s=65, facecolors="none", label="target boundary")
    q = result["obstacle_points"]
    if q:
        plt.scatter([p["pixel_xy"][0] for p in q], [p["pixel_xy"][1] for p in q], marker="x", s=60, label="obstacle boundary")
    q = result["placement_candidates"]
    if q:
        plt.scatter([p["pixel_xy"][0] for p in q], [p["pixel_xy"][1] for p in q], marker="+", s=90, label="placement candidate")
    plt.title("KALM Initial Hybrid Keypoint Proposal")
    plt.axis("off"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "keypoints_2d_visualization.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("=" * 60)
    print("KALM INITIAL KEYPOINT PROPOSAL")
    print("=" * 60)
    print(f"Grasp points         : {len(result['grasp_points'])}")
    print(f"Target boundary      : {len(result['target_boundary_points'])}")
    print(f"Obstacle points      : {len(result['obstacle_points'])}")
    print(f"Placement candidates : {len(result['placement_candidates'])}")
    print(f"JSON                 : {out / 'keypoints_2d.json'}")
    print(f"Visualization        : {out / 'keypoints_2d_visualization.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
