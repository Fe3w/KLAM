#!/usr/bin/env python3
"""
One-time Dobot screenshot keypoint test.

Purpose:
    Test the existing KALM 2D keypoint proposal pipeline on a Dobot/Isaac Sim
    screenshot without modifying the main KALM code.

Pipeline:
    image + existing SAM masks + placement ROI
        -> square crop
        -> DINOv2/FeatUp at 224x224
        -> existing propose_keypoints()
        -> map keypoints back to original image
        -> save JSON + visualization

Important:
    The mask JSON and placement JSON MUST belong to the same image passed via
    --image. Do not reuse masks from a different screenshot.

Example:
    python test_dobot_keypoints.py \
        --image data/dobot.jpg \
        --mask-json output/dobot_masks/masks.json \
        --placement-json output/dobot_masks/placement_region.json \
        --output output/dobot_keypoints
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from kalm.keypoint_predictor import DINOFeatureExtractor
from kalm.keypoint_proposal import ProposalConfig, propose_keypoints


def resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve absolute paths and paths stored relative to the metadata file."""
    p = Path(path_value)
    if p.is_absolute():
        return p

    candidates = [
        p,
        base_dir / p,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return the most useful path for the eventual error message.
    return (base_dir / p).resolve()


def load_mask_json(path: str) -> list[dict]:
    """
    Load masks.json.

    The saved mask_file can be absolute or relative to the JSON file.
    """
    json_path = Path(path).resolve()

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise RuntimeError(f"Expected a list in {json_path}")

    output = []

    for record in records:
        if not isinstance(record, dict):
            continue

        if "mask_file" not in record:
            raise RuntimeError(f"Missing mask_file in record: {record}")

        mask_path = resolve_path(record["mask_file"], json_path.parent)

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask file does not exist:\n  {mask_path}\n"
                f"Referenced from:\n  {json_path}"
            )

        item = dict(record)
        item["mask"] = np.load(mask_path).astype(bool)
        output.append(item)

    return output


def load_placement_mask(path: str | None) -> np.ndarray | None:
    """Load placement_region.json and its referenced binary mask."""
    if path is None:
        return None

    json_path = Path(path).resolve()

    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    mask_file = metadata.get("mask_file")
    if not mask_file:
        raise RuntimeError(f"No mask_file found in {json_path}")

    mask_path = resolve_path(mask_file, json_path.parent)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Placement mask does not exist:\n  {mask_path}"
        )

    return np.load(mask_path).astype(bool)


def center_square_crop(
    image: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    """
    Center-crop an HxWx3 image to a square.

    Returns:
        cropped_image, x0, y0, crop_size
    """
    h, w = image.shape[:2]

    crop_size = min(h, w)
    x0 = (w - crop_size) // 2
    y0 = (h - crop_size) // 2

    cropped = image[
        y0:y0 + crop_size,
        x0:x0 + crop_size,
    ].copy()

    return cropped, x0, y0, crop_size


def crop_mask(
    mask: np.ndarray,
    x0: int,
    y0: int,
    crop_size: int,
) -> np.ndarray:
    """Apply exactly the same spatial crop used for the RGB image."""
    return mask[
        y0:y0 + crop_size,
        x0:x0 + crop_size,
    ].copy()


def validate_same_shape(
    image: np.ndarray,
    masks: list[dict],
    placement_mask: np.ndarray | None,
) -> None:
    """Check that all masks belong to the supplied image."""
    h, w = image.shape[:2]

    for i, item in enumerate(masks):
        mask = np.asarray(item["mask"])
        if mask.shape != (h, w):
            raise RuntimeError(
                f"Mask {i} ({item.get('label', 'unknown')}) has shape "
                f"{mask.shape}, but image is {(h, w)}.\n"
                "This normally means the masks were generated from a "
                "different image."
            )

    if placement_mask is not None and placement_mask.shape != (h, w):
        raise RuntimeError(
            f"Placement mask has shape {placement_mask.shape}, "
            f"but image is {(h, w)}.\n"
            "Generate the placement mask from the same image."
        )


def crop_all_masks(
    masks: list[dict],
    placement_mask: np.ndarray | None,
    x0: int,
    y0: int,
    crop_size: int,
) -> tuple[list[dict], np.ndarray | None]:
    """Crop all semantic masks and placement ROI using identical coordinates."""
    cropped_masks = []

    for item in masks:
        new_item = dict(item)
        new_item["mask"] = crop_mask(
            np.asarray(item["mask"]).astype(bool),
            x0,
            y0,
            crop_size,
        )
        cropped_masks.append(new_item)

    cropped_placement = None
    if placement_mask is not None:
        cropped_placement = crop_mask(
            placement_mask,
            x0,
            y0,
            crop_size,
        )

    return cropped_masks, cropped_placement


def remap_point(point_xy: list[int | float], x0: int, y0: int) -> list[int]:
    """Map [x,y] from square-crop coordinates back to original image coordinates."""
    return [
        int(round(float(point_xy[0]) + x0)),
        int(round(float(point_xy[1]) + y0)),
    ]


def remap_result_to_original(
    result: dict,
    x0: int,
    y0: int,
    original_w: int,
    original_h: int,
) -> dict:
    """
    Convert all 2D keypoint coordinates from crop coordinates to original image
    coordinates while preserving the rest of the proposal metadata.
    """
    out = dict(result)

    out["crop"] = {
        "type": "center_square",
        "x0": int(x0),
        "y0": int(y0),
        "size": int(min(original_w, original_h)),
    }

    out["coordinate_convention"] = "pixel_xy=[x,y], origin=top-left"
    out["coordinate_space"] = "original_image"

    def remap_list(items: list[dict]) -> list[dict]:
        mapped = []

        for item in items:
            new_item = dict(item)
            if "pixel_xy" in new_item:
                new_item["pixel_xy_crop"] = list(new_item["pixel_xy"])
                new_item["pixel_xy"] = remap_point(
                    new_item["pixel_xy"],
                    x0,
                    y0,
                )

            if "pixel_yx" in new_item:
                new_item["pixel_yx_crop"] = list(new_item["pixel_yx"])
                xy = new_item["pixel_xy"]
                new_item["pixel_yx"] = [xy[1], xy[0]]

            mapped.append(new_item)

        return mapped

    for key in [
        "grasp_points",
        "target_boundary_points",
        "obstacle_points",
        "placement_candidates",
    ]:
        if key in out:
            out[key] = remap_list(out[key])

    out["image_width"] = int(original_w)
    out["image_height"] = int(original_h)

    return out


def run_dino(
    image_crop: np.ndarray,
    device: str,
) -> torch.Tensor:
    """
    Run FeatUp DINOv2 safely.

    The crop is square, and input_size=224 is exactly compatible with the
    14x14 DINOv2 patch size.
    """
    print(f"Loading DINO on {device}...")
    print("DINO input: 224 x 224")
    print("DINO patch size: 14 x 14")

    extractor = DINOFeatureExtractor(
        device=device,
        input_size=224,
        output_size=256,
    )

    try:
        with torch.inference_mode():
            features = extractor.extract_feature_from_image(image_crop)

        return features

    except RuntimeError as exc:
        message = str(exc).lower()

        if device == "cuda" and (
            "out of memory" in message
            or "cuda out of memory" in message
        ):
            print("\nCUDA OOM during DINO inference.")
            print("Retrying once on CPU for this one-time test...")

            del extractor
            gc.collect()
            torch.cuda.empty_cache()

            extractor = DINOFeatureExtractor(
                device="cpu",
                input_size=224,
                output_size=256,
            )

            with torch.inference_mode():
                features = extractor.extract_feature_from_image(
                    image_crop
                )

            return features

        raise


def save_visualization(
    image: np.ndarray,
    result: dict,
    output_path: Path,
) -> None:
    """Save the final keypoint visualization on the ORIGINAL image."""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.imshow(image)

    # Grasp
    grasp = result.get("grasp_points", [])
    if grasp:
        xs = [p["pixel_xy"][0] for p in grasp]
        ys = [p["pixel_xy"][1] for p in grasp]
        ax.scatter(
            xs,
            ys,
            marker="*",
            s=220,
            label="grasp",
        )

    # Target boundary
    target = result.get("target_boundary_points", [])
    if target:
        xs = [p["pixel_xy"][0] for p in target]
        ys = [p["pixel_xy"][1] for p in target]
        ax.scatter(
            xs,
            ys,
            marker="o",
            s=70,
            facecolors="none",
            linewidths=2,
            label="target boundary",
        )

    # Obstacles
    obstacles = result.get("obstacle_points", [])
    if obstacles:
        xs = [p["pixel_xy"][0] for p in obstacles]
        ys = [p["pixel_xy"][1] for p in obstacles]
        ax.scatter(
            xs,
            ys,
            marker="x",
            s=70,
            linewidths=2,
            label="obstacle boundary",
        )

    # Placement
    placement = result.get("placement_candidates", [])
    if placement:
        xs = [p["pixel_xy"][0] for p in placement]
        ys = [p["pixel_xy"][1] for p in placement]
        ax.scatter(
            xs,
            ys,
            marker="+",
            s=120,
            linewidths=2,
            label="placement candidate",
        )

    crop = result.get("crop", {})
    if crop:
        x0 = crop["x0"]
        y0 = crop["y0"]
        size = crop["size"]

        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                size,
                size,
                fill=False,
                linewidth=1.5,
                linestyle="--",
                label="DINO square crop",
            )
        )

    ax.set_title("Dobot One-Time KALM Keypoint Test")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_crop_visualization(
    image_crop: np.ndarray,
    crop_result: dict,
    output_path: Path,
) -> None:
    """Save crop-space visualization for debugging alignment."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_crop)

    for p in crop_result.get("grasp_points", []):
        x, y = p["pixel_xy"]
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=220,
            label="grasp",
        )

    target = crop_result.get("target_boundary_points", [])
    if target:
        ax.scatter(
            [p["pixel_xy"][0] for p in target],
            [p["pixel_xy"][1] for p in target],
            marker="o",
            s=70,
            facecolors="none",
            linewidths=2,
            label="target boundary",
        )

    obstacles = crop_result.get("obstacle_points", [])
    if obstacles:
        ax.scatter(
            [p["pixel_xy"][0] for p in obstacles],
            [p["pixel_xy"][1] for p in obstacles],
            marker="x",
            s=70,
            linewidths=2,
            label="obstacle boundary",
        )

    placement = crop_result.get("placement_candidates", [])
    if placement:
        ax.scatter(
            [p["pixel_xy"][0] for p in placement],
            [p["pixel_xy"][1] for p in placement],
            marker="+",
            s=120,
            linewidths=2,
            label="placement candidate",
        )

    ax.set_title("DINO Crop-Space Keypoints")
    ax.axis("off")
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time Dobot KALM keypoint test"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Dobot/Isaac Sim RGB screenshot",
    )

    parser.add_argument(
        "--mask-json",
        required=True,
        help="SAM masks.json generated from THIS image",
    )

    parser.add_argument(
        "--placement-json",
        required=True,
        help="placement_region.json generated from THIS image",
    )

    parser.add_argument(
        "--output",
        default="output/dobot_keypoints",
        help="Output directory",
    )

    parser.add_argument(
        "--n-target-points",
        type=int,
        default=6,
        help="Target boundary points",
    )

    parser.add_argument(
        "--n-obstacle-points",
        type=int,
        default=6,
        help="Obstacle boundary points per obstacle",
    )

    parser.add_argument(
        "--n-placement-points",
        type=int,
        default=5,
        help="Placement candidates",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force DINO to run on CPU",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------
    image = np.asarray(
        Image.open(args.image).convert("RGB")
    )

    original_h, original_w = image.shape[:2]

    print("=" * 70)
    print("DOBOT ONE-TIME KALM KEYPOINT TEST")
    print("=" * 70)
    print(f"Image           : {args.image}")
    print(f"Original size   : {original_w} x {original_h}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load semantic masks and placement ROI
    # ------------------------------------------------------------
    masks = load_mask_json(args.mask_json)
    placement_mask = load_placement_mask(args.placement_json)

    validate_same_shape(
        image,
        masks,
        placement_mask,
    )

    print(f"Loaded SAM masks: {len(masks)}")

    for item in masks:
        print(
            f"  - {item.get('label', 'unknown')} "
            f"[role={item.get('role', '')}, "
            f"downstream={item.get('downstream', '')}]"
        )

    # ------------------------------------------------------------
    # Square crop
    # ------------------------------------------------------------
    image_crop, x0, y0, crop_size = center_square_crop(image)

    print("\nSquare crop:")
    print(f"  crop size : {crop_size} x {crop_size}")
    print(f"  offset    : x={x0}, y={y0}")

    cropped_masks, cropped_placement = crop_all_masks(
        masks,
        placement_mask,
        x0,
        y0,
        crop_size,
    )

    # ------------------------------------------------------------
    # DINO
    # ------------------------------------------------------------
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dino = run_dino(
        image_crop,
        device,
    )

    print(f"DINO feature shape: {tuple(dino.shape)}")

    # ------------------------------------------------------------
    # Existing KALM keypoint proposal
    # ------------------------------------------------------------
    result_crop = propose_keypoints(
        image=image_crop,
        masks=cropped_masks,
        placement_mask=cropped_placement,
        dino_feature_map=dino,
        config=ProposalConfig(
            n_object_boundary_points=args.n_target_points,
            n_obstacle_boundary_points=args.n_obstacle_points,
            n_placement_candidates=args.n_placement_points,
        ),
    )

    # ------------------------------------------------------------
    # Map crop coordinates back to original image
    # ------------------------------------------------------------
    result = remap_result_to_original(
        result=result_crop,
        x0=x0,
        y0=y0,
        original_w=original_w,
        original_h=original_h,
    )

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------
    json_path = output_dir / "keypoints_2d.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ------------------------------------------------------------
    # Save visualizations
    # ------------------------------------------------------------
    original_vis = output_dir / "keypoints_2d_visualization.png"
    crop_vis = output_dir / "keypoints_crop_visualization.png"

    save_visualization(
        image=image,
        result=result,
        output_path=original_vis,
    )

    save_crop_visualization(
        image_crop=image_crop,
        crop_result=result_crop,
        output_path=crop_vis,
    )

    # Also save the crop itself for debugging.
    crop_path = output_dir / "dino_square_crop.png"
    Image.fromarray(image_crop).save(crop_path)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Grasp points          : {len(result['grasp_points'])}")
    print(f"Target boundary       : {len(result['target_boundary_points'])}")
    print(f"Obstacle points       : {len(result['obstacle_points'])}")
    print(f"Placement candidates  : {len(result['placement_candidates'])}")
    print(f"\nJSON                  : {json_path}")
    print(f"Original visualization: {original_vis}")
    print(f"Crop visualization    : {crop_vis}")
    print(f"DINO crop             : {crop_path}")
    print("=" * 70)
    print("Done.")
    print("This is only a 2D keypoint proposal test.")
    print("No robot control, depth, FPFH, IK, or temporal tracking is used.")


if __name__ == "__main__":
    main()
