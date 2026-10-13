import os
import json
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = "data/dobot.jpg"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
OUTPUT_PATH = "output/kimi_regions.json"

# ============================================================
# TASK
# ============================================================

TASK = input("Task: ").strip()

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are a robotics perception and task reasoning module controlling a
Dobot Magician robotic arm.

ROBOT INFORMATION:

- Robot: Dobot Magician
- End effector: suction gripper
- The robot grasps objects by creating suction on a suitable surface.
- The suction gripper therefore needs a reasonably flat, visible,
  accessible surface on the target object.
- Thin edges, curved surfaces, holes, handles, and highly irregular
  surfaces may be poor suction points unless the image clearly shows
  that they are suitable.
- The robot approaches the object from above in typical pick-and-place
  operations, so consider whether the target surface is accessible from above.
- Avoid proposing grasp points that are blocked by other objects.

Your job is to analyze the image and robot task and produce a structured
perception and task-reasoning result.

Determine:

1. TARGET OBJECT

   The object the robot must pick up or manipulate.

2. GRASP REGION

   The most suitable visible region on the target object for a suction
   gripper.

   Prefer:
   - flat surfaces
   - sufficiently large surfaces
   - visible surfaces
   - unobstructed surfaces
   - surfaces accessible from above

   Avoid:
   - thin edges
   - curved regions
   - object boundaries
   - holes
   - occluded regions
   - regions occupied by other objects

3. OBSTACLES

   Objects or regions that could interfere with:
   - approaching the target
   - grasping the target
   - lifting the target
   - moving the target
   - reaching the placement location

4. PLACEMENT GOAL

   ALWAYS provide a placement_goal for pick-and-place tasks.

   There are two possible placement modes:

   A. OBJECT-RELATIVE PLACEMENT

      If the task specifies a relation to another object, identify:
      - the reference object
      - the reference object's bounding box
      - the spatial relation
      - the approximate placement region

      Example:
      "Pick up the blue cube and place it to the right of the gray ball"

      reference_object = "gray ball"
      relation = "right_of"

   B. WORKSPACE-RELATIVE PLACEMENT

      If the task specifies only a workspace direction such as:
      - "place it on the right"
      - "place it on the left"
      - "place it above"
      - "place it below"

      interpret the direction relative to the image/workspace coordinate
      system.

      In this case:
      - reference_object = null
      - reference_bbox = null
      - use a workspace-relative relation
      - placement_region MUST be a clear, physically plausible free-space
        region in that part of the workspace

      Example:
      "Pick up the blue cube and place it on the right"

      relation = "workspace_right"

      placement_region = a clear floor region on the right side of the
      workspace where the cube can physically be placed.

   The placement_region must represent the final location of the target
   object, NOT the target object's current location.
   Possible relations include:
   - left_of
   - right_of
   - above
   - below
   - in_front_of
   - behind
   - near
   - inside
   - on_top_of

5. PLACEMENT ACCESSIBILITY

   Consider whether the target can physically be placed in the proposed
   destination without obvious collision with surrounding objects.

IMPORTANT:

For placement tasks, ALWAYS provide a placement_goal.

Return ONLY valid JSON.

Do not return markdown.

Do not wrap the JSON in ```.

Use this schema:

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
  "reference_object": null,
  "reference_description": null,
  "reference_bbox": null,
  "relation": "workspace_right",
  "placement_region": [x1, y1, x2, y2],
  "reason": "...",
  "confidence": 0.0
    }
}

RULES:

1. Coordinates are pixel coordinates with origin at the top-left.
2. Only identify objects actually visible in the image.
3. Do not invent objects.
4. The target must match the user's task.
5. The grasp_region must be a region ON the target object.
6. The grasp region should be appropriate for a suction gripper.
7. The reference_bbox must correspond to the object used in the
   placement relation.
8. placement_region represents the approximate final location of the
   target object, not the reference object.
9. For example, if the task is:
     "Pick up the blue cube and place it to the left of the gray ball"
   then:
     target = blue cube
     reference_object = gray ball
     relation = left_of
     placement_region = region where the blue cube should end up.
10. Do not confuse the target object with the placement reference object.
11. Reason about whether the target and placement area are accessible
    to a Dobot Magician with a suction gripper.
12. Prefer conservative, physically plausible regions over arbitrary
    regions.
13. When the task uses "on the right", "to the right", "on the left",
    "to the left", "above", or "below" without naming a reference object,
    interpret the relation relative to the visible workspace/image.

14. Workspace coordinates use the image coordinate system:
    - left = lower x
    - right = higher x
    - above = lower y
    - below = higher y

15. For workspace-relative placement, choose a clear floor/table region
    with enough area to contain the target object and maintain clearance
    from visible obstacles and workspace boundaries.
"""

# ============================================================
# API KEY / CLIENT
# ============================================================

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set.\n"
        "Run:\n"
        "export NVIDIA_API_KEY='YOUR_KEY'"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
)

# ============================================================
# LOAD IMAGE
# ============================================================

if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

with Image.open(IMAGE_PATH).convert("RGB") as image:
    width, height = image.size

mime_type, _ = mimetypes.guess_type(IMAGE_PATH)
if mime_type is None:
    mime_type = "image/jpeg"

image_bytes = Path(IMAGE_PATH).read_bytes()
encoded_image = base64.b64encode(image_bytes).decode("utf-8")
image_data_url = f"data:{mime_type};base64,{encoded_image}"

# ============================================================
# USER PROMPT
# ============================================================

user_prompt = f"""
Robot task:
{TASK}

Image dimensions:
width = {width}
height = {height}

Analyze the image using the robotics perception and task-reasoning
instructions provided in the system prompt.

Return the result using exactly the requested JSON schema.
"""

# ============================================================
# KIMI-K3 API CALL (streaming + bounded timeout)
# ============================================================

response = client.chat.completions.create(
    model=MODEL,
    max_tokens=6000,
    temperature=1,
    top_p=0.95,
    stream=True,
    timeout=120.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    },
                },
            ],
        },
    ],
)

# ============================================================
# GET + CLEAN OUTPUT
# ============================================================

print("\nWaiting for Nemotron 3 Ultra 550B A55B NVIDIA API response (streaming)...")
print("WARNING: NVIDIA currently lists this model as text-input-only. This file attempts the same image request to test the endpoint, but it is not expected to be a drop-in image-grounding replacement.")
print("Max tokens: 6000 | stream: on | read timeout: 120s")

chunks = []
reasoning_notice_printed = False
try:
    for chunk in response:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta

        # Some reasoning models emit reasoning_content before answer content.
        # Show a status notice only; do not print the reasoning trace itself.
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content and not reasoning_notice_printed:
            print("\\n[Model is reasoning; waiting for JSON answer...]\\n", flush=True)
            reasoning_notice_printed = True

        content = getattr(delta, "content", None)
        if content:
            print(content, end="", flush=True)
            chunks.append(content)
except Exception as exc:
    raise RuntimeError(
        f"Nemotron 3 Ultra 550B A55B streaming request failed or timed out: {exc}"
    ) from exc

raw_text = "".join(chunks).strip()

print("\n\n" + "=" * 60)
print("NEMOTRON 3 ULTRA 550B A55B REGION PROPOSAL")
print("=" * 60)
print(raw_text)

text = raw_text

# Remove accidental markdown fences.
if "```json" in text:
    text = text.split("```json", 1)[1]

if "```" in text:
    text = text.split("```", 1)[0]

text = text.strip()

# Extract the JSON object if there is surrounding text.
start = text.find("{")
end = text.rfind("}")

if start == -1 or end == -1:
    raise RuntimeError(
        "Nemotron 3 Ultra 550B A55B did not return a JSON object.\n\n"
        f"Response:\n{raw_text}"
    )

json_text = text[start:end + 1]

try:
    result = json.loads(json_text)
except json.JSONDecodeError as exc:
    raise RuntimeError(
        "Nemotron 3 Ultra 550B A55B returned invalid JSON.\n\n"
        f"Response:\n{raw_text}"
    ) from exc

# ============================================================
# SAVE
# ============================================================

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)
print(f"Model  : {MODEL}")
print(f"Image  : {IMAGE_PATH}")
print(f"Size   : {width} x {height}")
print(f"Output : {OUTPUT_PATH}")
print("=" * 60)
