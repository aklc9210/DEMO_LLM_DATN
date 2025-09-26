"""Response processing utilities."""

import json
from typing import Dict, Any

from .utils import get_logger

logger = get_logger("response_processor")


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text, handle cases where text is truncated."""
    text = text.strip()

    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No JSON found in response")

    brace_count = 0
    end_idx = -1

    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx == -1:
        logger.warning("JSON appears to be truncated, attempting to fix...")
        json_text = text[start_idx:]

        open_braces = json_text.count('{') - json_text.count('}')
        open_brackets = json_text.count('[') - json_text.count(']')

        if open_brackets > 0:
            json_text += ']' * open_brackets

        if open_braces > 0:
            json_text += '}' * open_braces

        return json_text

    return text[start_idx:end_idx]


def normalize_to_claude_like(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize different model responses to Claude-like format."""
    logger.info(f"normalize_to_claude_like input: {type(raw)} - {raw}")

    if isinstance(raw, dict) and "content" in raw:
        return raw

    txt = None
    if isinstance(raw, dict):
        logger.info(f"Raw is dict with keys: {list(raw.keys())}")

        # Nova messages-v1 format
        try:
            if "output" in raw and isinstance(raw["output"], dict):
                m = raw["output"].get("message")
                if isinstance(m, dict):
                    cont = m.get("content")
                    if isinstance(cont, list) and cont:
                        maybe_text = cont[0].get("text")
                        if isinstance(maybe_text, str) and maybe_text.strip():
                            txt = maybe_text.strip()
                            logger.info("Extracted Nova messages-v1 text.")
        except Exception:
            pass

        # Titan format
        if not txt:
            try:
                res = raw.get("results")
                if isinstance(res, list) and res and isinstance(res[0], dict):
                    ot = res[0].get("outputText")
                    if isinstance(ot, str) and ot.strip():
                        txt = ot.strip()
                        logger.info("Extracted Titan outputText.")
            except Exception:
                pass

        # Jamba/Nova text-only: outputText at root
        if not txt:
            ot = raw.get("outputText")
            if isinstance(ot, str) and ot.strip():
                txt = ot.strip()
                logger.info("Extracted root outputText.")

        # Llama: generation
        if not txt:
            gen = raw.get("generation")
            if isinstance(gen, str) and gen.strip():
                txt = gen.strip()
                logger.info("Extracted Llama generation.")

        # Other common keys
        if not txt:
            for key in ("result", "output", "completion", "text"):
                val = raw.get(key)
                if isinstance(val, str) and val.strip():
                    txt = val.strip()
                    logger.info(f"Extracted via key '{key}'.")
                    break

    if not txt:
        txt = json.dumps(raw, ensure_ascii=False) if raw is not None else ""
        logger.warning("No specific field matched; using JSON stringify fallback.")
        if not txt or txt.strip() in ("", "null", "{}"):
            raise ValueError("Model returned empty/unsupported structure")

    return {"content": [{"type": "text", "text": txt}]}