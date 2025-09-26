"""Model inference service."""

import time
import io
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from .utils import to_base64, get_logger
from .prompt_builder import (
    build_prompt, build_prompt_with_image, build_prompt_titan,
    build_prompt_llama, build_prompt_nova, build_prompt_nova_with_image
)
from .models import get_model_cost_estimates

logger = get_logger("inference")


def build_body_for_model(model_id: str, desc: str, temperature: float, max_tokens: int) -> dict:
    """Create prompt body for specific model."""
    logger.info(f"Building prompt for model: {model_id}")

    mid = (model_id or "").strip().lower()

    if 'nova' in mid:
        logger.info(f"Using Nova prompt builder for {model_id}")
        return build_prompt_nova(desc, temperature=temperature, max_tokens=max_tokens)

    if 'titan' in mid:
        logger.info(f"Using Titan prompt builder for {model_id}")
        return build_prompt_titan(desc, temperature=temperature, max_tokens=max_tokens)

    if 'llama' in mid:
        logger.info(f"Using Llama prompt builder for {model_id}")
        return build_prompt_llama(desc, temperature=temperature, max_tokens=max_tokens)

    logger.info(f"Using Claude prompt builder for {model_id}")
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

    raw = bedrock_client.invoke(model_id=model_id, body=body)
    latency = time.time() - t0

    logger.info(f"Raw response type: {type(raw)}")
    logger.info(f"Raw response: {raw}")

    if not raw:
        logger.error("Empty response from Bedrock!")
        raise RuntimeError("AWS Bedrock returned empty response")

    # Estimate metrics (simplified)
    tokens_in = len(desc.split()) * 1.3  # rough estimate
    from .parser import extract_text
    from .response_processor import normalize_to_claude_like
    response_text = extract_text(normalize_to_claude_like(raw))
    tokens_out = len(response_text.split()) * 1.3

    # Cost estimation
    cost_per_1k_in, cost_per_1k_out = get_model_cost_estimates(model_id)
    cost_est = (tokens_in/1000 * cost_per_1k_in) + (tokens_out/1000 * cost_per_1k_out)

    metrics = {
        "latency_s": round(latency, 2),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "cost_est_usd": round(cost_est, 6)
    }

    return raw, metrics