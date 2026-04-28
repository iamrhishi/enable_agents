"""
Prompts Blueprint — save and retrieve AI prompt history.
"""
import json
import os
from datetime import datetime
from uuid import uuid4

from flask import Blueprint, jsonify, request

prompts_bp = Blueprint("prompts", __name__)

_PROMPTS_FILE = os.environ.get(
    "PROMPTS_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "prompts.json"),
)


def _load_prompts() -> list:
    if os.path.exists(_PROMPTS_FILE):
        with open(_PROMPTS_FILE, "r") as f:
            return json.load(f)
    return []


def _save_prompts(prompts: list) -> None:
    os.makedirs(os.path.dirname(_PROMPTS_FILE), exist_ok=True)
    with open(_PROMPTS_FILE, "w") as f:
        json.dump(prompts, f, indent=2)


@prompts_bp.post("/save-prompt")
def save_prompt():
    data = request.get_json() or {}
    data["id"] = str(uuid4())
    data["timestamp"] = datetime.utcnow().isoformat()
    prompts = _load_prompts()
    prompts.append(data)
    _save_prompts(prompts)
    return jsonify({"message": "Prompt saved successfully", "id": data["id"]})


@prompts_bp.get("/previous-prompts")
def previous_prompts():
    return jsonify({"prompts": _load_prompts()})
