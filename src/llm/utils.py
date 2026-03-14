"""Utility functions for LLM responses."""

import json
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_response(response: str) -> dict | list:
    """Extract and parse JSON from an LLM response.

    Strategies tried in order:
    1. Direct json.loads on the full response
    2. Extract content from markdown code fences (```json ... ```)
    3. Find the first { ... } or [ ... ] block in the text
    4. Attempt to repair truncated JSON (close missing brackets)
    """
    text = response.strip()

    # 1. Try direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # 2. Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if not fence_match:
        # Handle truncated response where closing ``` is missing
        fence_match = re.search(r"```(?:json)?\s*\n(.*)", text, re.DOTALL)
    if fence_match:
        extracted = fence_match.group(1).strip()
        result = _try_parse(extracted)
        if result is not None:
            return result
        # Try repairing the fenced content if truncated
        result = _try_repair(extracted)
        if result is not None:
            logger.warning("JSON was truncated, repaired by closing brackets")
            return result

    # 3. Try finding the outermost { } or [ ] block
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        if start == -1:
            continue
        end = text.rfind(close_char)
        if end > start:
            result = _try_parse(text[start : end + 1])
            if result is not None:
                return result

    # 4. Last resort: try to repair from the first { or [
    for open_char in ("{", "["):
        start = text.find(open_char)
        if start != -1:
            result = _try_repair(text[start:])
            if result is not None:
                logger.warning("JSON was truncated, repaired by closing brackets")
                return result

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


def _try_parse(text: str) -> dict | list | None:
    """Attempt to parse JSON, return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_repair(text: str) -> dict | list | None:
    """Try to repair truncated JSON by closing unclosed brackets and quotes."""
    repaired = text.rstrip()
    # Strip trailing incomplete string value or key
    repaired = re.sub(r',\s*"[^"]*$', '', repaired)
    repaired = re.sub(r':\s*"[^"]*$', ': null', repaired)
    repaired = re.sub(r',\s*$', '', repaired)

    # Track actual nesting order to close brackets correctly
    stack = []
    in_string = False
    escape = False
    for ch in repaired:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append('}' if ch == '{' else ']')
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack:
        return None

    # Close in reverse nesting order
    closing = ''.join(reversed(stack))
    repaired += closing
    result = _try_parse(repaired)
    if result is not None:
        logger.warning(
            "JSON repair: closed %d unclosed bracket(s) (%s). "
            "Data after the truncation point is LOST.",
            len(stack), closing,
        )
    return result
