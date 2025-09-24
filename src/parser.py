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
        # fallback: có thể chuỗi chính là JSON nhưng không bắt đầu bằng '{'
        return t
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start:i+1]
                # kiểm tra nhanh có phải JSON hợp lệ không
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    break
    # nếu không tìm được theo cặp ngoặc, thử regex thô cuối cùng
    m = re.search(r"\{[\s\S]*\}", t)
    return m.group(0) if m else t

def extract_text(model_response: Dict[str, Any]) -> str:
    # Claude-like: {"content":[{"type":"text","text":"..."}]}
    content = model_response.get("content", [])
    if not content or content[0].get("type") != "text":
        raise ValueError("Unexpected response format: missing text content")
    text = content[0].get("text", "").strip()
    return text

def parse_and_validate(model_response: Dict[str, Any]) -> Dish:
    try:
        raw_text = extract_text(model_response)
        json_text = _extract_json_from_text(raw_text)
        data = json.loads(json_text)
        return Dish.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.exception("Failed to parse/validate: %s", e)
        # debug tiện lợi: log ra vài ký tự đầu
        logger.error("Raw text (head): %s", raw_text[:500] if 'raw_text' in locals() else "")
        raise
