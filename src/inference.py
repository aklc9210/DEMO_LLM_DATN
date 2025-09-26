"""Model inference service."""
import time
import io
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from .utils import to_base64, get_logger
from .prompt_builder import build_prompt as build_prompt
from .prompt_builder import (
    build_prompt, build_prompt_with_image, build_prompt_titan,
    build_prompt_llama, build_prompt_nova, build_prompt_nova_with_image,
)
from .models import get_model_cost_estimates

logger = get_logger("inference")


def build_body_for_model(model_id: str, desc: str, temperature: float, max_tokens: int) -> dict:
    logger.info(f"Building prompt for model: {model_id}")
    mid = (model_id or "").strip().lower()

    if 'nova' in mid:
        return build_prompt_nova(desc, temperature=temperature, max_tokens=max_tokens)

    if 'titan-text-lite' in mid:
        return build_prompt_titan(desc, temperature=temperature, max_tokens=max_tokens)

    if 'llama' in mid:
        return build_prompt_llama(desc, temperature=temperature, max_tokens=max_tokens)

    # Claude Anthropic mặc định
    return build_prompt(desc, temperature=temperature, max_tokens=max_tokens)


def invoke_model(
    bedrock_client,
    desc: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    img: Optional[Image.Image] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Invoke model and return (response, metrics).
    metrics contains: latency_s, tokens_in, tokens_out, cost_est_usd
    """
    if not bedrock_client or not model_id:
        raise RuntimeError("Bedrock client/model_id not ready")

    t0 = time.time()

    # Build request body
    if img is not None:
        buf = io.BytesIO()
        img_format = "PNG"
        mime = "image/png"
        img.save(buf, img_format)
        b64 = to_base64(buf.getvalue())

        if 'nova' in model_id.lower():
            body = build_prompt_nova_with_image(desc, b64, mime, temperature=temperature, max_tokens=max_tokens)
        else:
            body = build_prompt_with_image(desc, b64, mime, temperature=temperature, max_tokens=max_tokens)
    else:
        body = build_body_for_model(model_id, desc, temperature, max_tokens)

    # Call Bedrock: nhận (raw_json, headers)
    raw, hdrs = bedrock_client.invoke(model_id=model_id, body=body)
    latency = time.time() - t0

    if not raw:
        raise RuntimeError("AWS Bedrock returned empty response")

    # ---------- TOKEN THẬT ----------
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None

    # a) InvokeModel: token nằm ở HTTP headers
    if hdrs:
        try:
            tokens_in = int(hdrs.get("x-amzn-bedrock-input-token-count")
                            or hdrs.get("x-amzn-bedrock-input-tokens") or 0)
            tokens_out = int(hdrs.get("x-amzn-bedrock-output-token-count")
                             or hdrs.get("x-amzn-bedrock-output-tokens") or 0)
        except Exception:
            pass

    # b) Converse/ConverseStream: token nằm trong body.usage
    if (tokens_in is None or tokens_out is None) and isinstance(raw, dict):
        usage = raw.get("usage") or {}
        if isinstance(usage, dict):
            if tokens_in is None:
                tokens_in = usage.get("inputTokens") or usage.get("input_tokens")
            if tokens_out is None:
                tokens_out = usage.get("outputTokens") or usage.get("output_tokens")

    # c) Fallback: đếm input bằng CountTokens (không tốn phí)
    if not tokens_in:
        try:
            tokens_in = bedrock_client.count_tokens(model_id, body) or 0
        except Exception:
            tokens_in = 0

    tokens_in = int(tokens_in or 0)
    tokens_out = int(tokens_out or 0)

    # ---------- GIÁ /1K TOKENS (ĐỒNG BỘ TỪ AWS PRICING) ----------
    cost_in_1k, cost_out_1k = get_model_cost_estimates(model_id)
    cost_est = (tokens_in / 1000.0) * cost_in_1k + (tokens_out / 1000.0) * cost_out_1k

    metrics = {
        "latency_s": round(latency, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_est_usd": round(cost_est, 6),
    }
    return raw, metrics
