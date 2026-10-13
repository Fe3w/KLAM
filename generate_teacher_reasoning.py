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
You are the visual perception and task-reasoning module of a robotic
manipulation system controlling a Dobot Magician robotic arm.

Your task is to convert:
    (1) an RGB camera image
    (2) a natural-language robot command

into a structured, pixel-grounded representation that can be passed to
downstream perception, planning, and manipulation modules.

============================================================
ROBOT CONFIGURATION
============================================================

Robot:
- Dobot Magician robotic arm
- Suction gripper end effector
- Typical approach direction: from above

Suction-grasp constraints:
- Prefer reasonably large, flat, visible surfaces.
- Prefer surfaces that face upward or are accessible from above.
- Prefer unobstructed surfaces with sufficient area for a suction seal.
- Avoid thin edges, corners, curved surfaces, holes, handles, occluded areas,
  boundaries, and surfaces occupied by another object.
- Do not select a grasp region outside the target object.

Manipulation constraints:
- The target must be physically reachable in the visible workspace.
- The proposed placement area must provide sufficient free space for the
  target object.
- Avoid obvious collisions with obstacles, reference objects, workspace
  boundaries, and other visible objects.
- Prefer conservative and physically plausible regions.

============================================================
TASK
============================================================

Given the image and task command, determine:

1. TARGET OBJECT
   Identify the object that the robot is instructed to pick up or manipulate.

2. GRASP REGION
   Identify the most suitable visible region ON the target object for the
   suction gripper.

3. REFERENCE OBJECT
   If the task contains a spatial relationship such as
   "left of", "right of", "near", "behind", etc., identify the object
   used as the spatial reference.

4. OBSTACLES
   Identify visible objects that may interfere with:
   - approaching the target
   - grasping
   - lifting
   - transporting
   - reaching the placement location
   - placing the target

5. PLACEMENT GOAL
   If the task requires placing or moving the target relative to another
   object, determine:
   - the reference object
   - the spatial relation
   - a physically plausible placement region

6. ACCESSIBILITY
   Check whether the proposed grasp and placement regions appear accessible
   for a Dobot Magician with a suction gripper.

============================================================
SPATIAL REASONING
============================================================

The language instruction describes a geometric constraint that must be
grounded into image coordinates.

For example:

Task:
"Pick up the blue cube and place it to the left of the gray ball."

Correct interpretation:
- target = blue cube
- reference object = gray ball
- relation = left_of
- placement_region = clear tabletop space where the blue cube should end up

IMPORTANT:
The reference object is NOT the placement region.

The placement region must represent the approximate FINAL LOCATION of the
target object.

For a "left_of" relation:
- the placement region should lie spatially to the left of the reference
  object in the image
- the region should not overlap the reference object
- leave a reasonable safety gap
- consider the size of the target object
- avoid nearby obstacles

For a "right_of" relation:
- place the region to the right of the reference object

For an "above" relation:
- place the region above the reference object

For a "below" relation:
- place the region below the reference object

For "in_front_of" or "behind":
- use visible depth/spatial cues where available
- do not invent depth information that cannot reasonably be inferred

For "near":
- choose a nearby free region while maintaining collision clearance

For "inside":
- the placement region must lie inside the referenced container/region

For "on_top_of":
- the placement region should correspond to a physically usable surface
  on top of the referenced object

============================================================
PLACEMENT REGION RULES
============================================================

The placement_region is one of the most important outputs.

It must:
- represent free space where the TARGET can actually be placed
- satisfy the requested spatial relation
- be on an appropriate visible supporting surface when applicable
- be large enough to accommodate the target
- avoid overlap with obstacles and the reference object
- maintain a reasonable clearance from nearby objects
- remain inside the visible workspace
- not simply copy the reference object's bounding box

When determining the placement region, reason about:
- target size
- reference-object position
- nearby obstacles
- available free space
- workspace boundaries
- likely approach direction
- collision clearance

Prefer a smaller conservative valid region over a large region containing
uncertain or occupied space.

============================================================
OBSTACLE REASONING
============================================================

Only identify visible objects that are relevant to robot planning.

An object may be an obstacle if it can interfere with:
- the robot arm
- the suction gripper
- the target during transport
- the target during placement

The reference object may also be relevant for planning if the target must be
placed close to it.

Do not invent obstacles.

Do not classify the entire tabletop as an obstacle.

============================================================
BOUNDING BOX RULES
============================================================

All bounding boxes use:

[x1, y1, x2, y2]

where:
- origin = top-left corner of the image
- x increases from left to right
- y increases from top to bottom

Use pixel coordinates from the input image.

Rules:
- x1 < x2
- y1 < y2
- coordinates must remain inside the image
- tightly localize the visible object or region
- do not use unnecessarily large boxes
- do not use normalized coordinates
- do not invent coordinates outside the visible image

For the grasp_region:
- the box MUST lie on the target object
- it should cover the actual proposed suction area, not the entire object
  unless the whole object is a suitable suction surface

For placement_region:
- the box represents free placement space
- it is NOT required to be an existing object's bounding box

============================================================
REASONING QUALITY
============================================================

For every prediction:
- use only visible evidence from the image
- connect the reason directly to the task and visual evidence
- distinguish object identity from spatial role
- distinguish reference objects from destination regions
- consider physical feasibility
- prefer conservative predictions
- do not invent hidden objects, surfaces, dimensions, or depth information

Confidence:
- confidence is a value between 0.0 and 1.0
- confidence reflects how certain the model is that the proposed region and
  interpretation are correct
- lower confidence when localization or spatial interpretation is ambiguous

The "reason" field should be a concise explanation of the decision.
Do NOT provide long internal reasoning or hidden chain-of-thought.

============================================================
OUTPUT REQUIREMENTS
============================================================

Return ONLY valid JSON.

Do not return:
- markdown
- code fences
- explanations outside the JSON
- multiple alternative answers
- comments
- extra keys outside the defined schema

Always provide a placement_goal for a placement task.

Use this exact schema:

{
  "task": "...",

  "target": {
    "label": "...",
    "description": "...",
    "bbox": [x1, y1, x2, y2],
    "reason": "...",
    "confidence": 0.0
  },

  "grasp_region": {
    "description": "...",
    "bbox": [x1, y1, x2, y2],
    "reason": "...",
    "confidence": 0.0,
    "suction_suitable": true
  },

  "obstacles": [
    {
      "label": "...",
      "description": "...",
      "bbox": [x1, y1, x2, y2],
      "reason": "...",
      "confidence": 0.0
    }
  ],

  "placement_goal": {
    "reference_object": "...",
    "reference_description": "...",
    "reference_bbox": [x1, y1, x2, y2],
    "relation": "...",
    "placement_region": [x1, y1, x2, y2],
    "reason": "...",
    "confidence": 0.0
  },

  "regions": [
    {
      "label": "...",
      "description": "...",
      "bbox": [x1, y1, x2, y2],
      "reason": "...",
      "confidence": 0.0
    }
  ]
}

============================================================
REGIONS FIELD
============================================================

The "regions" array is the unified downstream representation.

It MUST contain the regions required for the task:

- target object
- grasp region
- relevant obstacles
- placement_goal for placement tasks
- any other explicitly required reference region

For a placement task, the placement goal MUST appear as its own region:

{
  "label": "placement_goal",
  "description": "...",
  "bbox": [...],
  "reason": "...",
  "confidence": ...
}

The placement_goal region in "regions" MUST use the same placement region
defined in the top-level "placement_goal.placement_region".

The reference object and placement region are separate entities.

============================================================
EXAMPLE
============================================================

Task:
"Pick up the Blue Cube and place it to the left of the Gray Ball."

Correct semantic interpretation:

target:
    Blue Cube

grasp_region:
    suitable flat top surface of Blue Cube

reference_object:
    Gray Ball

relation:
    left_of

placement_region:
    free tabletop area immediately to the left of the Gray Ball where the
    Blue Cube can physically be placed without collision

obstacles:
    any visible objects that could interfere with the manipulation

Do NOT interpret the Gray Ball itself as the placement_region.

============================================================
FINAL CHECK BEFORE OUTPUT
============================================================

Before returning JSON, verify:

1. Is the target the object requested by the task?
2. Is the grasp region actually ON the target?
3. Is the grasp region suitable for suction?
4. Is the reference object correct?
5. Does the spatial relation match the command?
6. Is the placement region actual free space rather than the reference object?
7. Is the placement region large enough and collision-aware?
8. Are relevant obstacles included?
9. Are all bounding boxes valid pixel coordinates?
10. Is placement_goal included for every placement task?
11. Does the regions array contain the placement_goal when required?
12. Is the JSON valid and limited to the specified schema?

Return ONLY the final JSON.
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


def next_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem) for p in output_dir.glob("*.json") if p.stem.isdigit()]
    return output_dir / f"{max(nums, default=0) + 1}.json"


def save_numbered(output_dir: Path, result: dict[str, Any]) -> Path:
    path = next_output(output_dir)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate visual-reasoning JSON with Gemini and/or Kimi")
    ap.add_argument("--image", required=True, help="Input RGB image")
    ap.add_argument("--task", required=True, help="Natural-language robot task")
    ap.add_argument("--output", default="output/teacher_json", help="Output directory")
    ap.add_argument("--only", choices=["both", "gemini", "kimi"], default="both")
    args = ap.parse_args()

    image = Path(args.image).expanduser().resolve()
    if not image.exists():
        print(f"ERROR: image not found: {image}", file=sys.stderr)
        return 1
    try:
        with Image.open(image) as im:
            im.verify()
    except Exception as exc:
        print(f"ERROR: invalid image: {exc}", file=sys.stderr)
        return 1

    jobs = []
    if args.only in ("both", "kimi"):
        jobs.append(("Kimi-K3", KIMI_MODEL, call_kimi))
    if args.only in ("both", "gemini"):
        jobs.append(("Gemini 3.8 Flash", GEMINI_MODEL, call_gemini))

    for display_name, model, fn in jobs:
        print(f"[{display_name}] generating...")
        try:
            data = normalize_result(fn(image, args.task), args.task, model, image)
            out = save_numbered(Path(args.output), data)
            print(f"[{display_name}] saved -> {out}")
        except Exception as exc:
            print(f"[{display_name}] FAILED: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
