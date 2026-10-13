from __future__ import annotations

import argparse
import json
from pathlib import Path

from kalm.mask_proposal import (
    MaskProposal,
    save_masks,
    visualize_masks,
)


def normalize_regions(data):
    if "regions" in data:
        return data["regions"]

    regions = []

    if "target" in data:
        target = data["target"]

        regions.append({
            "label": target.get("label", "target_object"),
            "role": "target_object",
            "description": target.get("description", ""),
            "bbox": target["bbox"],
            "reason": target.get("reason", ""),
            "confidence": target.get("confidence", 0.0),
        })

    if "grasp_region" in data:
        grasp = data["grasp_region"]

        regions.append({
            "label": "grasp_region",
            "role": "grasp_region",
            "description": grasp.get("description", ""),
            "bbox": grasp["bbox"],
            "reason": grasp.get("reason", ""),
            "confidence": grasp.get("confidence", 0.0),
        })

    for obstacle in data.get("obstacles", []):
        regions.append({
            "label": obstacle.get("label", "obstacle"),
            "role": "obstacle",
            "description": obstacle.get("description", ""),
            "bbox": obstacle["bbox"],
            "reason": obstacle.get("reason", ""),
            "confidence": obstacle.get("confidence", 0.0),
        })

    return regions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SAM masks from KALM region proposals"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input image",
    )

    parser.add_argument(
        "--regions",
        required=True,
        help="Region proposal JSON",
    )

    parser.add_argument(
        "--output",
        default="output/masks",
        help="Output directory",
    )

    parser.add_argument(
        "--model",
        default="sam_b.pt",
        help="SAM model checkpoint",
    )

    args = parser.parse_args()

    print("Loading region JSON...")

    with open(args.regions, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = normalize_regions(data)

    print(f"Found {len(regions)} regions.")

    for region in regions:
        print(
            f"  {region['label']}: "
            f"{region['bbox']}"
        )

    if not regions:
        print("No regions found.")
        return

    # Convert normalized coordinates -> pixel coordinates.
    # Your 005_kimi.json currently contains values in the 0..1 range.
    with open(args.image, "rb") as f:
        from PIL import Image
        image = Image.open(f)
        width, height = image.size

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

    print("\nPixel coordinates sent to SAM:")

    for region in regions:
        print(
            f"  {region['label']}: "
            f"{region['bbox']}"
        )

    print("\nLoading SAM model...")

    mask_generator = MaskProposal(args.model)

    masks = mask_generator.generate_masks(
        image_path=args.image,
        regions=regions,
    )

    save_masks(
        masks,
        args.output,
    )

    visualize_masks(
        image_path=args.image,
        masks=masks,
        output_path=str(
            Path(args.output) / "mask_visualization.png"
        ),
    )


if __name__ == "__main__":
    main()