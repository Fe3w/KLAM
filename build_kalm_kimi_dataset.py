#!/usr/bin/env python3
"""
Build a KALM distillation dataset from RGB images on a flat workspace.

For each image in <dataset>/pending_images:
  1) Kimi inspects the image and generates N pick/place questions.
  2) Each question is sent to Kimi again for the KALM grounding JSON.
  3) The existing scripts.main_kalm_mask_verification SAM stage is run.
  4) One JSON record is saved per question.
  5) After all N questions succeed, the image is moved to used_images/.
  6) manifest.jsonl gets one clean entry per image/question pair.

Rerunning is safe for completed question records: existing successful
records are skipped. An image remains in pending_images/ until all questions
finish successfully.

NOTE ON BACKEND
----------------
This script calls the official Kimi API Platform using its OpenAI-compatible
Chat Completions endpoint. It does NOT use Kimi API or OpenRouter.

Use:
  KIMI_API_KEY=sk-...
  KIMI_BASE_URL=https://api.moonshot.ai/v1
  KIMI_MODEL=kimi-k3
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

# ---------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------
# Kimi API publishes a *specific* model string for each hosted model.
# Do NOT assume it equals the Moonshot marketing name.
#
# Verify the exact ID on your account with:
#   curl -sS https://api.moonshot.ai/v1/models \
#     -H "Authorization: Bearer $KIMI_API_KEY" | grep -i kimi
#
# Set via env var KIMI_MODEL or --model to override.
DEFAULT_KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k3")

# Kimi K3 supports low/high/max reasoning effort.
# Set KIMI_REASONING_EFFORT="" to omit the field.
REASONING_EFFORT = os.getenv("KIMI_REASONING_EFFORT", "high").strip()

KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
KIMI_URL = f"{KIMI_BASE_URL}/chat/completions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QUESTION_SYSTEM = r"""
You are a task-generation module for a Dobot Magician robotic manipulation dataset.

The robot works over a flat tabletop/workspace with a suction gripper and
approaches objects from above.

Look carefully at the RGB image and generate exactly the requested number of
natural-language pick-and-place tasks using ONLY visible objects.

Rules:
- Each task has one visible object to pick up.
- Each task has a clearly specified destination/reference object OR a clear
  workspace-relative destination.
- Prefer: left_of, right_of, above, below, near, workspace_left,
  workspace_right, workspace_above, workspace_below.
- Use color + shape when useful to disambiguate objects.
- Do not use the Dobot arm as a target/reference for placement questions.
- Do not invent hidden objects or containers.
- Avoid impossible, blocked, or obviously colliding placements.
- Avoid duplicate or nearly duplicate questions.
- Do NOT answer the tasks; generate the task questions only.

Return ONLY JSON:
{
  "questions": ["...", "..."]
}
"""

GROUNDING_SYSTEM = r"""
You are a visual reasoning module for a Dobot Magician robotic manipulation system.

The robot uses a suction gripper and approaches objects from above.

Analyze the RGB image and the robot task.

Identify:
1. Target object
2. Grasp region suitable for suction
3. Relevant obstacles
4. Reference object
5. Placement goal / placement region

For every region provide label, description, bbox [x1,y1,x2,y2], reason,
and confidence. Coordinates are pixel coordinates with origin at top-left.

For placement tasks:
- reference_object is the object defining the relation
- placement_region is the FREE workspace region where the target should end up
- never use the reference object itself as the placement region
- keep placement physically plausible and away from obstacles

The workspace is a flat tabletop. Only use visible evidence.
Return ONLY valid JSON.
"""


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.I | re.S)
    if m:
        text = m.group(1).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError(f"No JSON object found:\n{text}")
    return json.loads(text[a:b + 1])


def get_api_key() -> str:
    """Return a cleaned Kimi API key."""
    key = os.getenv("KIMI_API_KEY")
    if not key:
        raise RuntimeError(
            "KIMI_API_KEY is not set.\n"
            "Get your key from https://platform.kimi.ai/console/api-keys, then run:\n"
            "    export KIMI_API_KEY='sk-...'"
        )
    key = key.strip().strip('"').strip("'")
    if not key.startswith("sk-"):
        raise RuntimeError(
            "KIMI_API_KEY does not look like a Kimi API key.\n"
            f"  length = {len(key)}\n"
            f"  prefix = {key[:6]!r}\n"
            "Kimi Platform API keys normally start with 'sk-'."
        )
    return key


# ---------------------------------------------------------------------
# KIMI API CALL
# ---------------------------------------------------------------------

def call_kimi(
    image: Path,
    system: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: str,
) -> dict[str, Any]:
    key = get_api_key()
    with Image.open(image) as im:
        w, h = im.size

    payload: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}\n\nImage dimensions: width={w}, height={h}"},
                    {"type": "image_url", "image_url": {"url": image_url(image)}},
                ],
            },
        ],
    }
    if REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT

    try:
        r = requests.post(
            KIMI_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Kimi API request error: {e}") from e

    if not r.ok:
        raise RuntimeError(
            f"Kimi API HTTP {r.status_code}: {r.text}\n"
            f"Endpoint   : {KIMI_URL}\n"
            f"Model used : {model}\n"
        )

    body = r.json()
    try:
        content = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected Kimi response: {body}") from e
    return parse_json(content)


# ---------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------

def generate_questions(image: Path, n: int, model: str) -> list[str]:
    prompt = (
        f"Generate exactly {n} distinct feasible pick-and-place questions for this image. "
        "Do not answer them."
    )
    last = None
    for attempt in range(3):
        try:
            out = call_kimi(image, QUESTION_SYSTEM, prompt, 2500, 0.4, model)
            qs = out.get("questions")
            if not isinstance(qs, list):
                raise ValueError("Missing questions list")
            clean, seen = [], set()
            for q in qs:
                if not isinstance(q, str):
                    continue
                q = re.sub(r"\s+", " ", q).strip()
                key = q.lower().rstrip(".?!")
                if len(q) >= 15 and key not in seen:
                    seen.add(key)
                    clean.append(q)
            if len(clean) == n:
                return clean
            raise ValueError(f"Expected {n} unique questions, got {len(clean)}")
        except Exception as e:
            last = e
            prompt = (
                f"Generate exactly {n} unique feasible pick-and-place questions. "
                f"Previous attempt was invalid: {e}. Use only visible tabletop objects."
            )
    raise RuntimeError(f"Question generation failed: {last}")


def ground_question(image: Path, question: str, model: str) -> dict[str, Any]:
    prompt = f"""
Robot task:
{question}

Ground this task in the image and return this JSON structure:
{{
  "task": "...",
  "image_width": 0,
  "image_height": 0,
  "target": {{"label":"...","description":"...","bbox":[x1,y1,x2,y2],"reason":"...","confidence":0.0}},
  "grasp_region": {{"description":"...","bbox":[x1,y1,x2,y2],"reason":"...","confidence":0.0,"suction_suitable":true}},
  "obstacles": [{{"label":"...","description":"...","bbox":[x1,y1,x2,y2],"reason":"...","confidence":0.0}}],
  "placement_goal": {{"reference_object":"...","reference_description":"...","reference_bbox":[x1,y1,x2,y2],"relation":"...","placement_region":[x1,y1,x2,y2],"reason":"...","confidence":0.0}},
  "regions": []
}}
Return ONLY JSON.
"""
    last = None
    for attempt in range(3):
        try:
            out = call_kimi(image, GROUNDING_SYSTEM, prompt, 9000, 0.2, model)
            required = ["target", "grasp_region", "obstacles", "placement_goal"]
            missing = [k for k in required if k not in out]
            if missing:
                raise ValueError(f"Missing grounding fields: {missing}")
            out["task"] = question
            out.setdefault("regions", [])
            return out
        except Exception as e:
            last = e
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Kimi grounding failed after 3 attempts: {last}")


def run_sam(repo_root: Path, image: Path, kimi_json: Path, out_dir: Path, model: str) -> None:
    cmd = [
        sys.executable, "-m", "scripts.main_kalm_mask_verification",
        "--image", str(image), "--regions", str(kimi_json),
        "--output", str(out_dir), "--model", model,
    ]
    print("  SAM -> running")
    p = subprocess.run(cmd, cwd=str(repo_root), text=True)
    if p.returncode != 0:
        raise RuntimeError(f"SAM failed with exit code {p.returncode}")
    if not (out_dir / "semantic_verification.png").exists():
        raise RuntimeError("SAM finished without semantic_verification.png")


def pending_images(path: Path) -> list[Path]:
    path.mkdir(parents=True, exist_ok=True)
    return sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda x: str(x).lower(),
    )


def next_image_id(records: Path) -> str:
    ids = []
    for p in records.glob("*_questions.json"):
        m = re.match(r"^(\d+)_questions\.json$", p.name)
        if m:
            ids.append(int(m.group(1)))
    return f"{max(ids, default=0)+1:06d}"


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_image(
    repo: Path,
    root: Path,
    image: Path,
    image_id: str,
    n: int,
    sam_model: str,
    kimi_model: str,
    state: dict[str, Any],
) -> bool:
    records = root / "records"
    sam_root = root / "sam"
    used = root / "used_images"
    state_path = root / "state.json"
    manifest = root / "manifest.jsonl"
    h = sha256(image)
    image_state = state["images"].setdefault(
        h,
        {
            "image_id": image_id,
            "source_name": image.name,
            "status": "processing",
            "started_at_utc": now(),
        },
    )
    image_state["questions_requested"] = n
    save_json(state_path, state)

    qplan_path = records / f"{image_id}_questions.json"
    qplan = load_json(qplan_path, {})
    if (
        isinstance(qplan, dict)
        and isinstance(qplan.get("questions"), list)
        and len(qplan["questions"]) == n
    ):
        questions = qplan["questions"]
        print("Using existing question plan")
    else:
        print("Generating questions with Kimi...")
        questions = generate_questions(image, n, kimi_model)
        save_json(
            qplan_path,
            {
                "image_id": image_id,
                "image_sha256": h,
                "image_pending": f"pending_images/{image.name}",
                "question_count": n,
                "questions": questions,
                "teacher_model": kimi_model,
                "created_at_utc": now(),
            },
        )

    for i, question in enumerate(questions, 1):
        rec_path = records / f"{image_id}_q{i:02d}.json"
        old = load_json(rec_path, {})
        if isinstance(old, dict) and old.get("status") == "complete":
            print(f"Q{i}: already complete, skip")
            continue

        sample_dir = sam_root / image_id / f"q{i:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        kimi_path = sample_dir / "kimi.json"
        print(f"\n[{image_id} Q{i:02d}] {question}")
        try:
            answer = ground_question(image, question, kimi_model)
            save_json(kimi_path, answer)
            run_sam(repo, image, kimi_path, sample_dir, sam_model)
            record = {
                "sample_id": f"{image_id}_q{i:02d}",
                "image_id": image_id,
                "question_id": f"q{i:02d}",
                "image": f"pending_images/{image.name}",
                "image_sha256": h,
                "question": question,
                "teacher_model": kimi_model,
                "teacher_output": answer,
                "sam": {
                    "dir": str(sample_dir.relative_to(root)),
                    "kimi_json": str(kimi_path.relative_to(root)),
                    "masks_json": str((sample_dir / "masks.json").relative_to(root))
                    if (sample_dir / "masks.json").exists() else None,
                    "placement_region": str((sample_dir / "placement_region.npy").relative_to(root))
                    if (sample_dir / "placement_region.npy").exists() else None,
                    "semantic_verification": str(
                        (sample_dir / "semantic_verification.png").relative_to(root)
                    ),
                },
                "status": "complete",
                "created_at_utc": now(),
            }
            save_json(rec_path, record)
            image_state["completed_questions"] = i
            save_json(state_path, state)
            print(f"[{image_id} Q{i:02d}] complete")
        except Exception as e:
            save_json(
                rec_path,
                {
                    "sample_id": f"{image_id}_q{i:02d}",
                    "image_id": image_id,
                    "question_id": f"q{i:02d}",
                    "image": f"pending_images/{image.name}",
                    "question": question,
                    "status": "failed",
                    "error": str(e),
                    "failed_at_utc": now(),
                },
            )
            image_state["status"] = "partial_failure"
            save_json(state_path, state)
            print(f"[{image_id} Q{i:02d}] FAILED: {e}", file=sys.stderr)
            return False

    # Move only after every question has succeeded.
    for i in range(1, n + 1):
        if load_json(records / f"{image_id}_q{i:02d}.json", {}).get("status") != "complete":
            return False

    used.mkdir(parents=True, exist_ok=True)
    dest = used / image.name
    k = 1
    while dest.exists():
        dest = used / f"{image.stem}_{k}{image.suffix}"
        k += 1
    shutil.move(str(image), str(dest))
    final_rel = str(dest.relative_to(root))

    for i in range(1, n + 1):
        rec_path = records / f"{image_id}_q{i:02d}.json"
        rec = load_json(rec_path, {})
        rec["image"] = final_rel
        save_json(rec_path, rec)
        append_jsonl(
            manifest,
            {
                "sample_id": rec["sample_id"],
                "image_id": image_id,
                "question_id": rec["question_id"],
                "image": final_rel,
                "question": rec["question"],
                "answer_json": str(rec_path.relative_to(root)),
                "sam_dir": rec["sam"]["dir"],
                "teacher_model": kimi_model,
            },
        )

    image_state.update(
        {
            "status": "complete",
            "used_image": final_rel,
            "completed_questions": n,
            "completed_at_utc": now(),
        }
    )
    save_json(state_path, state)
    print(f"\nIMAGE {image_id} COMPLETE -> {final_rel}")
    return True


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/kalm_kimi_dataset")
    ap.add_argument("--questions-per-image", type=int, default=6)
    ap.add_argument("--sam-model", default="sam_b.pt")
    ap.add_argument(
        "--kimi-model",
        default=DEFAULT_KIMI_MODEL,
        help="Kimi model ID (default: %(default)s). Example: kimi-k3",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()
    if not 5 <= args.questions_per_image <= 8:
        ap.error("--questions-per-image must be between 5 and 8")

    # Validate API key up front so we fail before touching the dataset.
    get_api_key()

    root = Path(args.dataset).expanduser().resolve()
    for d in ["pending_images", "used_images", "records", "sam"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    repo = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root else Path.cwd().resolve()
    )
    if not ((repo / "scripts").is_dir() and (repo / "kalm").is_dir()):
        raise RuntimeError(
            f"Could not find KALM repo at {repo}. Use --repo-root /home/abhishek/KALM"
        )

    state_path = root / "state.json"
    state = load_json(
        state_path,
        {"dataset_version": 1, "created_at_utc": now(), "images": {}},
    )
    images = pending_images(root / "pending_images")
    if args.limit:
        images = images[: args.limit]

    print("=" * 70)
    print("KALM KIMI API + SAM DATASET BUILDER")
    print("=" * 70)
    print(f"Dataset          : {root}")
    print(f"Pending images   : {len(images)}")
    print(f"Questions/image  : {args.questions_per_image}")
    print(f"Teacher          : {args.kimi_model}")
    print(f"Kimi API         : {KIMI_URL}")
    print(f"Reasoning effort : {REASONING_EFFORT or '(omitted)'}")
    print(f"SAM              : {args.sam_model}")
    print("=" * 70)

    next_id = int(next_image_id(root / "records"))
    done = 0
    for offset, image in enumerate(images):
        image_id = f"{next_id + offset:06d}"
        print(f"\n### {image_id}: {image}")
        if process_image(
            repo,
            root,
            image,
            image_id,
            args.questions_per_image,
            args.sam_model,
            args.kimi_model,
            state,
        ):
            done += 1
    print(f"\nFinished: {done}/{len(images)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())