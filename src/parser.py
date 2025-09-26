from typing import Any, Dict
from pydantic import ValidationError
from .schema import Dish
from .utils import get_logger
import json
import re

logger = get_logger("parser")

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # loại bỏ ```json ... ``` hoặc ``` ...
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    return t

def _extract_json_from_text(text: str) -> str:
    """
    Cố gắng tìm khối JSON { ... } đầu tiên hợp lệ trong chuỗi.
    Duyệt theo dấu ngoặc nhọn để bắt cặp.
    """
    t = _strip_code_fences(text)
    start = t.find("{")
    if start == -1:
        return t
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    break

    m = re.search(r"\{[\s\S]*\}", t)
    return m.group(0) if m else t

def extract_text(model_response: Dict[str, Any]) -> str:
    # Claude-like: {"content":[{"type":"text","text":"..."}]}
    content = model_response.get("content", [])
    if content and isinstance(content, list) and len(content) > 0:
        if content[0].get("type") == "text":
            return content[0].get("text", "").strip()
    
    # Titan format: {"results":[{"outputText":"..."}]}
    results = model_response.get("results", [])
    if results and isinstance(results, list) and len(results) > 0:
        output_text = results[0].get("outputText", "")
        if output_text:
            return output_text.strip()
    
    # Nova format: {"output":{"message":{"content":[{"text":"..."}]}}}
    output = model_response.get("output", {})
    if output:
        message = output.get("message", {})
        if message:
            nova_content = message.get("content", [])
            if nova_content and isinstance(nova_content, list) and len(nova_content) > 0:
                text_content = nova_content[0].get("text", "")
                if text_content:
                    return text_content.strip()
    
    # Llama format: {"generation":"..."}
    generation = model_response.get("generation", "")
    if generation:
        return generation.strip()
    
    # Fallback: try to find any text field
    for key in ["text", "response", "answer", "result"]:
        if key in model_response:
            return str(model_response[key]).strip()
    
    logger.error(f"Cannot extract text from response: {model_response}")
    raise ValueError("Unexpected response format: no recognizable text content")

def parse_and_validate(model_response: Dict[str, Any]) -> Dish:
    try:
        raw_text = extract_text(model_response)
        json_text = _extract_json_from_text(raw_text)
        data = json.loads(json_text)
        return Dish.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.exception("Failed to parse/validate: %s", e)

        logger.error("Raw text (head): %s", raw_text[:500] if 'raw_text' in locals() else "")
        raise
