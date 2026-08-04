"""Defensive extraction and schema validation for model completions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from reasonforge.schemas import ParseResult, StructuredSolution

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def completion_to_text(completion: Any) -> str:
    """Normalize strings and TRL standard/conversational completion structures."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        return content if isinstance(content, str) else ""
    if isinstance(completion, Sequence) and not isinstance(completion, (bytes, bytearray)):
        for item in completion:
            if isinstance(item, Mapping) and isinstance(item.get("content"), str):
                return str(item["content"])
        if len(completion) == 1 and isinstance(completion[0], str):
            return completion[0]
    return ""


def _decode_first_object(text: str) -> tuple[dict[str, object] | None, bool, str | None]:
    stripped = text.strip()
    fence = _FENCE_RE.fullmatch(stripped)
    candidate = fence.group(1).strip() if fence else stripped
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonstandard_constant,
    )
    try:
        value, end = decoder.raw_decode(candidate)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        # A small amount of prose around JSON is common; extract one object but
        # mark it non-exact so formatting rewards cannot be gamed with chatter.
        start = candidate.find("{")
        if start < 0:
            return None, False, f"invalid JSON: {exc}"
        try:
            value, relative_end = decoder.raw_decode(candidate[start:])
            end = start + relative_end
        except (json.JSONDecodeError, ValueError) as nested:
            return None, False, f"invalid JSON: {nested}"
        exact = not candidate[:start].strip() and not candidate[end:].strip()
    else:
        exact = fence is None and not candidate[end:].strip()
    if not isinstance(value, dict):
        return None, exact, "top-level JSON value must be an object"
    return value, exact, None


def parse_completion(completion: Any, *, max_chars: int = 4_000) -> ParseResult:
    """Parse an untrusted completion into the strict solution schema.

    The parser never executes generated content. It accepts a bare JSON object or
    a single fenced JSON object, and records whether surrounding prose existed.
    """
    text = completion_to_text(completion)
    if not text.strip():
        return ParseResult(raw_text=text, error="completion is empty")
    if len(text) > max_chars:
        return ParseResult(raw_text=text[:max_chars], error="completion exceeds size limit")
    data, exact, error = _decode_first_object(text)
    if data is None:
        return ParseResult(raw_text=text, exact_json=exact, error=error)
    try:
        solution = StructuredSolution.model_validate(data)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "root"
        return ParseResult(
            raw_text=text,
            json_valid=True,
            exact_json=exact,
            data=data,
            error=f"schema error at {location}: {first.get('msg', 'invalid value')}",
        )
    return ParseResult(
        raw_text=text,
        json_valid=True,
        schema_valid=True,
        exact_json=exact,
        data=data,
        solution=solution,
    )
