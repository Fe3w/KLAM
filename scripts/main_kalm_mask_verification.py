from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

from kalm.mask_proposal import MaskProposal, save_masks


def normalize_regions(data: dict) -> list[dict]:
    """
    Convert Kimi/KALM JSON into the canonical region format expected by SAM.

    Kimi may return a schema containing target, grasp_region, obstacles,
    placement_goal, and an empty regions list. An empty regions list must not
    hide the actual semantic fields.
    """
    explicit_regions = data.get("regions")

    # Use an explicit regions list only when it is actually populated.
    if isinstance(explicit_regions, list) and explicit_regions:
        normalized: list[dict] = []

        for region in explicit_regions:
            if not isinstance(region, dict):
                continue

            bbox = region.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            item = dict(region)

            if not item.get("role"):
                label = str(item.get("label", "")).strip().lower()
                if "grasp" in label:
                    item["role"] = "grasp_region"
                elif "placement" in label:
                    item["role"] = "placement_goal"
                else:
                    item["role"] = "other"

            normalized.append(item)

        if normalized:
            return normalized

    # Otherwise construct canonical regions from the richer Kimi schema.
    regions: list[dict] = []

    target = data.get("target")
    if isinstance(target, dict):
        bbox = target.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            regions.append({
                "label": target.get("label", "target_object"),
                "role": "target_object",
                "description": target.get("description", ""),
                "bbox": bbox,
                "reason": target.get("reason", ""),
                "confidence": target.get("confidence", 0.0),
            })

    grasp = data.get("grasp_region")
    if isinstance(grasp, dict):
        bbox = grasp.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            regions.append({
                "label": "grasp_region",
                "role": "grasp_region",
                "description": grasp.get("description", ""),
                "bbox": bbox,
                "reason": grasp.get("reason", ""),
                "confidence": grasp.get("confidence", 0.0),
            })

    obstacles = data.get("obstacles", [])
    if isinstance(obstacles, list):
        for obstacle in obstacles:
            if not isinstance(obstacle, dict):
                continue
            bbox = obstacle.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            regions.append({
                "label": obstacle.get("label", "obstacle"),
                "role": "obstacle",
                "description": obstacle.get("description", ""),
                "bbox": bbox,
                "reason": obstacle.get("reason", ""),
                "confidence": obstacle.get("confidence", 0.0),
            })

    # The placement_region is a workspace ROI, not a SAM object. The
    # placement reference object can still be passed to SAM.
    placement_goal = data.get("placement_goal")
    if isinstance(placement_goal, dict):
        reference_object = placement_goal.get("reference_object")
        reference_bbox = placement_goal.get("reference_bbox")

        if (
            reference_object
            and isinstance(reference_bbox, list)
            and len(reference_bbox) == 4
        ):
            existing_labels = {
                str(region.get("label", "")).strip().lower()
                for region in regions
            }

            if str(reference_object).strip().lower() not in existing_labels:
                regions.append({
                    "label": reference_object,
                    "role": "reference_object",
                    "description": placement_goal.get(
                        "reference_description", ""
                    ),
                    "bbox": reference_bbox,
                    "reason": "Placement reference object.",
                    "confidence": placement_goal.get("confidence", 0.0),
                })

    return regions


def convert_boxes_to_pixels(
    regions: list[dict],
    width: int,
    height: int,
) -> None:
    """Convert 0..1 boxes to pixels while leaving pixel boxes unchanged."""
    for region in regions:
        bbox = region["bbox"]

        if all(0.0 <= float(v) <= 1.0 for v in bbox):
            x1, y1, x2, y2 = bbox
            region["bbox"] = [
                int(round(x1 * width)),
                int(round(y1 * height)),
                int(round(x2 * width)),
                int(round(y2 * height)),
            ]
        else:
            region["bbox"] = [
                int(round(float(v))) for v in bbox
            ]

        x1, y1, x2, y2 = region["bbox"]
        region["bbox"] = [
            max(0, min(x1, width - 1)),
            max(0, min(y1, height - 1)),
            max(0, min(x2, width - 1)),
            max(0, min(y2, height - 1)),
        ]


def save_placement_region(
    placement_goal: dict,
    width: int,
    height: int,
    output_dir: Path,
) -> np.ndarray | None:
    """Create a binary workspace ROI from Kimi's placement bbox."""
    bbox = placement_goal.get("placement_region")

    if not isinstance(bbox, list) or len(bbox) != 4:
        print("No valid placement_region found.")
        return None

    x1, y1, x2, y2 = [
        int(round(float(v))) for v in bbox
    ]

    if all(0.0 <= float(v) <= 1.0 for v in bbox):
        x1 = int(round(bbox[0] * width))
        y1 = int(round(bbox[1] * height))
        x2 = int(round(bbox[2] * width))
        y2 = int(round(bbox[3] * height))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        print("Invalid placement_region.")
        return None

    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2 + 1, x1:x2 + 1] = True

    npy_path = output_dir / "placement_region.npy"
    png_path = output_dir / "placement_region.png"
    json_path = output_dir / "placement_region.json"

    np.save(npy_path, mask)
    Image.fromarray(
        (mask.astype(np.uint8) * 255)
    ).save(png_path)

    metadata = {
        "relation": placement_goal.get("relation"),
        "reference_object": placement_goal.get("reference_object"),
        "reference_bbox": placement_goal.get("reference_bbox"),
        "placement_region": [x1, y1, x2, y2],
        "reason": placement_goal.get("reason", ""),
        "confidence": placement_goal.get("confidence", 0.0),
        "mask_file": str(npy_path),
        "mask_png": str(png_path),
        "mask_shape": [height, width],
        "mask_dtype": str(mask.dtype),
        "type": "workspace_roi",
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nPlacement ROI:")
    print(f"  Relation : {metadata['relation']}")
    print(f"  BBox     : {metadata['placement_region']}")
    print(f"  Saved    : {npy_path}")
    print(f"  Saved    : {png_path}")

    return mask


def make_verification_visualization(
    image_path: str,
    sam_results: list[dict],
    placement_mask: np.ndarray | None,
    placement_goal: dict | None,
    output_path: Path,
) -> None:
    """Create one image showing target, grasp, obstacles, reference and goal."""
    image = np.asarray(
        Image.open(image_path).convert("RGB")
    )

    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    role_alpha = {
        "target_object": 0.45,
        "grasp_region": 0.55,
        "obstacle": 0.35,
        "reference_object": 0.35,
    }

    for item in sam_results:
        mask = item["mask"]
        role = item.get("downstream", "other")

        if role == "dino":
            label = item["label"]
            if "grasp" in label.lower():
                display_role = "GRASP"
            else:
                display_role = "TARGET / REFERENCE"
        elif role == "planning":
            display_role = "OBSTACLE / PLANNING"
        else:
            display_role = role.upper()

        ax.imshow(
            mask,
            alpha=role_alpha.get(item.get("role", ""), 0.30),
        )

        x1, y1, x2, y2 = item["bbox"]

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
                edgecolor="white",
            )
        )

        ax.text(
            x1,
            max(0, y1 - 4),
            f"{item['label']} [{display_role}]",
            fontsize=10,
            color="white",
            bbox={
                "facecolor": "black",
                "alpha": 0.65,
                "pad": 2,
            },
        )

    if placement_mask is not None and placement_goal is not None:
        ys, xs = np.where(placement_mask)

        if len(xs):
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            ax.imshow(
                np.ma.masked_where(
                    ~placement_mask,
                    np.ones_like(placement_mask, dtype=float),
                ),
                alpha=0.20,
                cmap="Greens",
            )

            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=3,
                    edgecolor="lime",
                )
            )

            relation = placement_goal.get("relation", "placement")
            ax.text(
                x1,
                min(height - 5, y2 + 14),
                f"PLACEMENT ROI [{relation}]",
                fontsize=10,
                color="lime",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.70,
                    "pad": 2,
                },
            )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("KALM Semantic Region Verification")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Kimi JSON -> SAM object masks + placement ROI + "
            "single verification visualization"
        )
    )
    parser.add_argument("--image", required=True, help="Input RGB image")
    parser.add_argument(
        "--regions",
        required=True,
        help="Kimi/KALM reasoning JSON",
    )
    parser.add_argument(
        "--output",
        default="output/kimi_masks",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        default="sam_b.pt",
        help="SAM model checkpoint",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading JSON...")
    with open(args.regions, "r", encoding="utf-8") as f:
        data = json.load(f)

    with Image.open(args.image).convert("RGB") as image:
        width, height = image.size

    regions = normalize_regions(data)
    convert_boxes_to_pixels(regions, width, height)

    print(f"Found {len(regions)} SAM regions.")

    for region in regions:
        print(
            f"  {region['label']} "
            f"[{region['role']}] : {region['bbox']}"
        )

    if not regions:
        raise RuntimeError("No object regions found in JSON.")

    print("\nLoading SAM...")
    mask_generator = MaskProposal(args.model)

    sam_results = mask_generator.generate_masks(
        image_path=args.image,
        regions=regions,
    )

    save_masks(sam_results, args.output)

    placement_goal = data.get("placement_goal")
    placement_mask = None

    if placement_goal:
        placement_mask = save_placement_region(
            placement_goal,
            width,
            height,
            output_dir,
        )

    verification_path = output_dir / "semantic_verification.png"

    make_verification_visualization(
        image_path=args.image,
        sam_results=sam_results,
        placement_mask=placement_mask,
        placement_goal=placement_goal,
        output_path=verification_path,
    )

    print("\nVerification output:")
    print(f"  {verification_path}")
    print("\nPipeline:")
    print("  Kimi JSON")
    print("     -> SAM masks for objects/regions")
    print("     -> placement ROI from placement_region")
    print("     -> semantic_verification.png")
    print("\nNote: placement ROI is not a SAM segmentation.")
    print("It is the workspace region proposed by Kimi.")


if __name__ == "__main__":
    main()
