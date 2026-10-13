#!/usr/bin/env python3
"""Generate numbered visual-reasoning JSON files from Gemini 3.8 Flash and Kimi-K3."""
from __future__ import annotations
import argparse, base64, json, mimetypes, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import requests
except ImportError:
    requests = None

GEMINI_MODEL = "gemini-3.8-flash"
KIMI_MODEL = "moonshotai/kimi-k3"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string"},
        "target": {
            "type": "object", "properties": {
                "label": {"type": "string"}, "description": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "reason": {"type": "string"}, "confidence": {"type": "number"}
            }, "required": ["label", "description", "bbox", "reason", "confidence"]
        },
        "grasp_region": {
            "type": "object", "properties": {
                "description": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "reason": {"type": "string"}, "confidence": {"type": "number"}
            }, "required": ["description", "bbox", "reason", "confidence"]
        },
        "obstacles": {
            "type": "array", "items": {"type": "object", "properties": {
                "label": {"type": "string"}, "description": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "reason": {"type": "string"}, "confidence": {"type": "number"}
            }, "required": ["label", "description", "bbox", "reason", "confidence"]}
        },
        "placement_goal": {
            "type": "object", "properties": {
                "reference_object": {"type": "string"}, "reference_description": {"type": "string"},
                "reference_bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "relation": {"type": "string"},
                "placement_region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "reason": {"type": "string"}, "confidence": {"type": "number"}
            }, "required": ["reference_object", "reference_description", "reference_bbox", "relation", "placement_region", "reason", "confidence"]
        },
        "regions": {
            "type": "array", "items": {"type": "object", "properties": {
                "label": {"type": "string"}, "description": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                "reason": {"type": "string"}, "confidence": {"type": "number"}
            }, "required": ["label", "description", "bbox", "reason", "confidence"]}
        }
    },
    "required": ["task", "target", "grasp_region", "obstacles", "placement_goal", "regions"]
}

SYSTEM_PROMPT = """
You are the visual reasoning module of a robotic manipulation system.
Given an RGB camera image and a natural-language task, ground the task in the image.
Return ONLY valid JSON matching the requested schema.

Reason about:
- target: object the robot must manipulate
- grasp_region: visible surface suitable for suction grasp
- obstacles: objects relevant to collision avoidance
- placement_goal: actual free-space region where the target should end up
- regions: flat list containing all useful grounded regions

For a task such as 'pick up the blue cube and place it to the left of the gray ball':
- gray ball = reference object
- relation = left_of
- placement_region = clear tabletop area where the cube can physically be placed
- DO NOT use the reference object itself as the placement region

Use pixel coordinates [x1, y1, x2, y2]. Confidence is 0..1.
Keep reasoning concise and based on visible evidence.
The regions list MUST include the target, grasp region, relevant obstacles/reference objects,
and placement_goal when a placement instruction exists.
"""


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[a:b + 1])


def normalize_result(result: dict[str, Any], task: str, model: str, image: Path) -> dict[str, Any]:
    result["task"] = result.get("task") or task
    regions = result.get("regions") if isinstance(result.get("regions"), list) else []
    pg = result.get("placement_goal")
    if isinstance(pg, dict) and isinstance(pg.get("placement_region"), list) and len(pg["placement_region"]) == 4:
        exists = any(isinstance(r, dict) and str(r.get("label", "")).lower().strip() == "placement_goal" for r in regions)
        if not exists:
            regions.append({
                "label": "placement_goal",
                "description": f"Final placement area {pg.get('relation', '')} {pg.get('reference_object', 'reference object')}".strip(),
                "bbox": pg["placement_region"],
                "reason": pg.get("reason", ""),
                "confidence": pg.get("confidence", 0.0),
            })
    result["regions"] = regions
    result["_metadata"] = {
        "model": model,
        "image": str(image),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return result


def call_gemini(image: Path, task: str) -> dict[str, Any]:
    if genai is None or types is None:
        raise RuntimeError("Missing google-genai. Install with: pip install google-genai")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)
    with Image.open(image) as im:
        prompt = SYSTEM_PROMPT + "\n\nTASK COMMAND:\n" + task
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, im.copy()],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SCHEMA,
                temperature=0.2,
                max_output_tokens=8192,
            ),
        )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")
    return parse_json(response.text)


def call_kimi(image: Path, task: str) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("Missing requests. Install with: pip install requests")
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    payload = {
        "model": KIMI_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT + "\n\nTASK COMMAND:\n" + task + "\n\nReturn ONLY JSON."},
            {"type": "image_url", "image_url": {"url": image_data_url(image)}}
        ]}],
        "max_tokens": 16384,
        "temperature": 0.2,
        "reasoning_effort": "max",
        "stream": False,
    }
    r = requests.post(NVIDIA_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=300)
    if not r.ok:
        raise RuntimeError(f"Kimi API HTTP {r.status_code}: {r.text}")
    body = r.json()
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Kimi response: {body}") from exc
    return parse_json(text)



def next_scene_id(images_dir: Path) -> int:
    """Return the next 1-based zero-padded scene id."""
    images_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in images_dir.iterdir():
        if p.is_file() and p.stem.isdigit():
            ids.append(int(p.stem))
    return max(ids, default=0) + 1


def copy_image(image: Path, images_dir: Path, scene_id: int) -> tuple[Path, str]:
    """
    Copy the source image into the dataset.

    Returns:
        absolute destination path,
        dataset-relative image path used inside JSON files.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    suffix = image.suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    destination = images_dir / f"{scene_id:03d}{suffix}"
    destination.write_bytes(image.read_bytes())

    # Paths stored in JSON are relative to the split directory.
    relative = f"{destination.parent.name}/images/{destination.name}"
    return destination, relative


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_train_annotation(
    annotation_dir: Path,
    scene_id: int,
    image_rel_path: str,
    result: dict[str, Any],
) -> Path:
    """
    Save the teacher output as a training annotation.

    The model name is deliberately NOT included in the JSON. The filename
    identifies which teacher produced the annotation for human bookkeeping.
    """
    annotation = dict(result)

    annotation["image"] = image_rel_path

    # Keep task as the task shown to the student model.
    # _metadata from the API stage is removed because it is not a training
    # target and should not teach the student about the teacher.
    annotation.pop("_metadata", None)
    annotation.pop("model", None)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    teacher_name = (
        "kimi"
        if result.get("_teacher_model") == KIMI_MODEL
        else "gemini"
    )
    path = annotation_dir / f"{scene_id:03d}_{teacher_name}.json"
    write_json(path, annotation)
    return path


def write_test_input(
    test_input_dir: Path,
    scene_id: int,
    image_rel_path: str,
    task: str,
) -> Path:
    """
    Save ONLY the inference inputs for the held-out test split.

    The student model will receive image + task and generate the JSON itself.
    """
    data = {
        "image": image_rel_path,
        "task": task,
    }
    path = test_input_dir / f"{scene_id:03d}.json"
    write_json(path, data)
    return path


def build_dataset_dirs(root: Path, split: str) -> dict[str, Path]:
    split_root = root / split
    dirs = {
        "root": split_root,
        "images": split_root / "images",
        "annotations": split_root / "annotations",
        "inputs": split_root / "inputs",
    }

    if split == "train":
        dirs["annotations"].mkdir(parents=True, exist_ok=True)
        dirs["images"].mkdir(parents=True, exist_ok=True)
    else:
        dirs["inputs"].mkdir(parents=True, exist_ok=True)
        dirs["images"].mkdir(parents=True, exist_ok=True)

    return dirs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create a robotics visual-reasoning train/test dataset. "
            "Train mode copies the image and generates Gemini/Kimi teacher "
            "annotations. Test mode copies the image and creates an input "
            "JSON containing only image path + task."
        )
    )
    ap.add_argument("--image", required=True, help="Input RGB image")
    ap.add_argument("--task", required=True, help="Natural-language robot task")
    ap.add_argument(
        "--dataset",
        default="output/vr_dataset",
        help="Dataset root directory",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Create training data or held-out test input data",
    )
    ap.add_argument(
        "--only",
        choices=["both", "gemini", "kimi"],
        default="both",
        help="In train mode, run both teachers or only one",
    )
    args = ap.parse_args()

    image = Path(args.image).expanduser().resolve()
    dataset_root = Path(args.dataset).expanduser().resolve()

    if not image.exists():
        print(f"ERROR: image not found: {image}", file=sys.stderr)
        return 1

    try:
        with Image.open(image) as im:
            im.verify()
    except Exception as exc:
        print(f"ERROR: invalid image: {exc}", file=sys.stderr)
        return 1

    dirs = build_dataset_dirs(dataset_root, args.split)
    scene_id = next_scene_id(dirs["images"])
    copied_image, image_rel_path = copy_image(image, dirs["images"], scene_id)

    print(f"\nDataset split : {args.split}")
    print(f"Scene ID      : {scene_id:03d}")
    print(f"Image         : {copied_image}")

    # ------------------------------------------------------------
    # TEST MODE
    # ------------------------------------------------------------
    # The test input intentionally contains only:
    #   image + task
    #
    # No teacher output is stored in the test input file.
    # ------------------------------------------------------------
    if args.split == "test":
        test_json = write_test_input(
            dirs["inputs"],
            scene_id,
            image_rel_path,
            args.task,
        )
        print(f"Test input    : {test_json}")
        print("\nTest input JSON contains only:")
        print(json.dumps({
            "image": image_rel_path,
            "task": args.task,
        }, indent=2))
        return 0

    # ------------------------------------------------------------
    # TRAIN MODE
    # ------------------------------------------------------------
    jobs = []
    if args.only in ("both", "kimi"):
        jobs.append((KIMI_MODEL, call_kimi))
    if args.only in ("both", "gemini"):
        jobs.append((GEMINI_MODEL, call_gemini))

    successful = 0

    for model_name, fn in jobs:
        display_name = "Kimi-K3" if model_name == KIMI_MODEL else "Gemini 3.8 Flash"
        print(f"\n[{display_name}] generating teacher annotation...")

        try:
            data = fn(copied_image, args.task)

            # Normalize before removing metadata.
            data = normalize_result(
                data,
                args.task,
                model_name,
                copied_image,
            )

            # Temporary internal marker used only to choose the output filename.
            data["_teacher_model"] = model_name

            annotation = dict(data)
            annotation["image"] = image_rel_path
            annotation.pop("_metadata", None)
            annotation.pop("_teacher_model", None)

            teacher_tag = "kimi" if model_name == KIMI_MODEL else "gemini"
            annotation_path = (
                dirs["annotations"] /
                f"{scene_id:03d}_{teacher_tag}.json"
            )
            write_json(annotation_path, annotation)

            print(f"[{display_name}] saved -> {annotation_path}")
            successful += 1

        except Exception as exc:
            print(f"[{display_name}] FAILED: {exc}", file=sys.stderr)

    if successful == 0:
        print(
            "\nERROR: No teacher annotation was generated. "
            "The image was copied, but no training label was created.",
            file=sys.stderr,
        )
        return 1

    print("\nTraining sample created:")
    print(f"  Image:       {copied_image}")
    print(f"  Annotations: {dirs['annotations']}")
    print(
        "\nThe image appears ONCE in train/images, while each teacher "
        "produces a separate annotation for the same scene."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
