"""Azure ML init/run adapter for the pinned Clef decision model on GPU, base or fine-tuned.

Azure calls init once per worker, then run with a JSON string containing state
and one native route choice question, plus an optional skill question. Importing this module never loads a model.
"""

# Azure's script loader omits sys.modules registration; Pydantic needs eager annotations.

import json
import os
from threading import Lock
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from routing_lab import clef
from routing_lab.taxonomy import SKILLS, TAXONOMIES, taxonomy_for_routes, taxonomy_routes

ROUTES = taxonomy_routes("coding-v1")
VARIANT = os.environ.get("CLEF_VARIANT", "flash")
DEVICE = os.environ.get("CLEF_DEVICE", "cuda")  # Azure keeps the default; a container on a CPU-only host can set "cpu" (slow)
# CLEF_FINETUNED=true serves the registered fine-tuned export and refuses to fall back to the base model.
FINETUNED = os.environ.get("CLEF_FINETUNED", "").lower() == "true"
MODEL = MappingProxyType(clef.model_info(VARIANT))
MAX_REQUEST_BYTES = 524_288
MAX_STATE_CHARS = 131_072
Description = Annotated[str, Field(min_length=1, max_length=2_048)]


class RouteQuestion(BaseModel):
    """Accept only a bounded native choice over the complete route taxonomy."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice"]
    instructions: str = Field(min_length=1, max_length=4_096)
    criteria: dict[str, Description] = Field(
        min_length=min(len(taxonomy_routes(key)) for key in TAXONOMIES),
        max_length=max(len(taxonomy_routes(key)) for key in TAXONOMIES),
    )

    @field_validator("instructions")
    @classmethod
    def nonblank_instructions(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Instructions must be nonblank")
        return value

    @field_validator("criteria")
    @classmethod
    def complete_routes(cls, value: dict[str, str]) -> dict[str, str]:
        taxonomy_for_routes(value)
        if any(not item.strip() for item in value.values()):
            raise ValueError("Criteria descriptions must be nonblank")
        return value


class SkillQuestion(BaseModel):
    """The optional second head: which kind of work the request involves, over the fixed skill list."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice"]
    instructions: str = Field(min_length=1, max_length=4_096)
    criteria: dict[str, Description] = Field(min_length=len(SKILLS), max_length=len(SKILLS))

    @field_validator("criteria")
    @classmethod
    def complete_skills(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != set(SKILLS) or any(not item.strip() for item in value.values()):
            raise ValueError("Criteria must describe exactly the supported skills")
        return value

    @field_validator("instructions")
    @classmethod
    def nonblank_instructions(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Instructions must be nonblank")
        return value


class Questions(BaseModel):
    """The route question, plus at most one optional skill question; no other heads are permitted."""

    model_config = ConfigDict(extra="forbid", strict=True)
    route: RouteQuestion
    skill: SkillQuestion | None = None


class Request(BaseModel):
    """No batch input, media, decoding options, or extra fields. ``max_length`` is the only
    model override; it is bounded by the ceiling this deployment's GPU memory was sized for,
    and omitting it keeps the release's own 16,384-token default."""

    model_config = ConfigDict(extra="forbid", strict=True)
    state: str = Field(min_length=1, max_length=MAX_STATE_CHARS)
    questions: Questions
    max_length: int | None = Field(default=None, ge=512, le=clef.MAX_LENGTH_CEILING)

    @field_validator("state")
    @classmethod
    def nonblank_state(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("State must be nonblank")
        return value


_runtime: clef.ClefRuntime | None = None
_fine_tuned: dict[str, Any] | None = None
_init_lock = Lock()
_infer_lock = Lock()


def init() -> None:
    """Load the pinned Clef release on the GPU, once per worker.

    Only Azure startup should invoke this. Startup requires outbound Hugging Face access;
    failure prevents the worker becoming ready.
    """
    global _runtime, _fine_tuned
    with _init_lock:
        if _runtime is None and FINETUNED:
            root = os.environ.get("AZUREML_MODEL_DIR")
            if not root:
                raise RuntimeError("CLEF_FINETUNED needs a registered model mounted at AZUREML_MODEL_DIR")
            export = clef.find_export(Path(root))
            _fine_tuned = clef.finetune_provenance(export)
            _runtime = clef.load(VARIANT, device=DEVICE, snapshot=export, calibrated=True)
        elif _runtime is None:
            _runtime = clef.load(VARIANT, device=DEVICE)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous duplicate JSON fields, including repeated route labels."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _error(status: int, code: str, message: str) -> Any:
    from azureml_inference_server_http.api.aml_response import AMLResponse

    return AMLResponse({"error": {"code": code, "message": message}}, status, json_str=True)


def run(raw_data: str) -> Any:
    """Validate one JSON request and preserve native answers and usage.

    Returns:
        A JSON-serializable native response with server-owned experiment_model
        metadata, or an AMLResponse with a sanitized 400/413/500/503 error.
        Request contents and validation/model exception details are never logged.
    """
    try:
        if not isinstance(raw_data, str):
            raise ValueError("Expected JSON text")
        if len(raw_data) > MAX_REQUEST_BYTES or len(raw_data.encode("utf-8")) > MAX_REQUEST_BYTES:
            return _error(413, "request_too_large", f"Request exceeds {MAX_REQUEST_BYTES} UTF-8 bytes.")
        request = Request.model_validate(json.loads(raw_data, object_pairs_hook=_unique_object))
    except (ValueError, TypeError, RecursionError):
        return _error(400, "invalid_request", "Expected bounded state and one supported complete route catalog.")

    if _runtime is None:
        return _error(503, "not_ready", "Model has not initialized.")
    try:
        questions = request.questions.model_dump(exclude_none=True)
        with _infer_lock:
            native = clef.decide(
                _runtime, request.state, questions,
                max_length=request.max_length or clef.DEFAULT_MAX_LENGTH,
            )
        heads = {"route": request.questions.route.criteria}
        if request.questions.skill is not None:
            heads["skill"] = request.questions.skill.criteria
        if set(native["answers"]) != set(heads):
            raise ValueError("Model answered a different set of questions")
        for name, options in heads.items():
            answer = native["answers"][name]
            probabilities = answer["probabilities"]
            if (
                answer["choice"] not in options or not isinstance(probabilities, dict)
                or set(probabilities) != set(options)
                or any(not 0 <= value <= 1 for value in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > 0.001
                or probabilities[answer["choice"]] < max(probabilities.values()) - 0.000001
            ):
                raise ValueError("Model returned a mismatched distribution")
        response = {
            "answers": native["answers"],
            "usage": native["usage"],
            "experiment_model": {**MODEL, "fine_tuned": dict(_fine_tuned)} if _fine_tuned else dict(MODEL),
        }
        # Keep serialization failures inside the sanitized error boundary.
        json.dumps(response, allow_nan=False)
        return response
    except (RuntimeError, ValueError, TypeError, KeyError, MemoryError):
        return _error(500, "inference_failed", "Model inference failed.")
