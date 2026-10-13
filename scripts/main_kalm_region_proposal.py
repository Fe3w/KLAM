from __future__ import annotations

import argparse
import json

from kalm.region_proposal import (
    get_region_proposals,
    save_region_proposals,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KALM image -> Gemini reasoning -> region proposals"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input image path",
    )

    parser.add_argument(
        "--task",
        required=True,
        help="Robot task/instruction",
    )

    parser.add_argument(
        "--output",
        default="output/test_regions.json",
        help="Output JSON file",
    )

    parser.add_argument(
    "--model",
    default="google/gemma-4-31b-it:free",
    help="Gemma/OpenRouter model",
    )

    args = parser.parse_args()

    print("\n========================================")
    print(" KALM REGION PROPOSAL")
    print("========================================")
    print(f"Image : {args.image}")
    print(f"Task  : {args.task}")
    print(f"Model : {args.model}")
    print("========================================\n")

    result = get_region_proposals(
        image_path=args.image,
        task=args.task,
        model=args.model,
    )

    save_region_proposals(
        result,
        args.output,
    )

    print("\n========== REASONING ==========")

    print(f"Task: {result['task']}")
    print(
        f"Image size: "
        f"{result['image_width']} x "
        f"{result['image_height']}"
    )

    print("\nRegions:")

    for i, region in enumerate(result["regions"]):
        print(f"\n[{i}] {region['label']}")
        print(f"Description: {region['description']}")
        print(f"BBox:        {region['bbox']}")
        print(f"Reason:      {region['reason']}")
        print(f"Confidence:  {region['confidence']}")

    print("\n================================")
    print(f"Saved to: {args.output}")
    print("================================\n")


if __name__ == "__main__":
    main()