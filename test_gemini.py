import os
import json
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = "data/test.jpg"
OUTPUT_PATH = "output/gemini_regions.json"

# ============================================================
# MODEL SELECTION
# ============================================================

choice = input(
    "Select Gemini model:\n"
    "1. Gemini 3.8 Flash\n"
    "2. Gemini 2.5 Flash\n"
    "Enter choice: "
).strip()

match choice:
    case "1":
        MODEL = "gemini-3.8-flash"
    case "2":
        MODEL = "gemini-2.5-flash"
    case _:
        print("Invalid choice")
        raise SystemExit(1)

# ============================================================
# TASK
# ============================================================

TASK = input("Task: ").strip()

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
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
   If the task requires placing or moving the target relative to another
   object, identify:
   - the reference object
   - the required spatial relation
   - the approximate placement region

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
    "reference_object": "...",
    "reference_description": "...",
    "reference_bbox": [x1, y1, x2, y2],
    "relation": "...",
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
"""

# ============================================================
# API KEY / CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Run:\n"
        "export GEMINI_API_KEY='YOUR_KEY'"
    )

client = genai.Client(api_key=api_key)

# ============================================================
# LOAD IMAGE
# ============================================================

if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size

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
# GEMINI API CALL
# ============================================================

response = client.models.generate_content(
    model=MODEL,
    contents=[user_prompt, image],
    config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
)

# ============================================================
# GET + CLEAN OUTPUT
# ============================================================

raw_text = (response.text or "").strip()

print("=" * 60)
print("GEMINI REGION PROPOSAL")
print("=" * 60)
print(raw_text)

text = raw_text

# Remove accidental markdown fences even if the model ignores
# the "Do not return markdown" instruction.
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
        "Gemini did not return a JSON object.\n\n"
        f"Response:\n{raw_text}"
    )

json_text = text[start:end + 2]

try:
    result = json.loads(json_text)
except json.JSONDecodeError as exc:
    raise RuntimeError(
        "Gemini returned invalid JSON.\n\n"
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
