"""Results display components."""

import json
import streamlit as st
from typing import Dict, Any

from ..parser import parse_and_validate, extract_text
from ..response_processor import normalize_to_claude_like, extract_json_from_text
from ..schema import Dish


def render_result(raw_response: Dict[str, Any], metrics: Dict[str, Any], model_name: str):
    """Render result with 2 columns layout."""
    try:
        with st.expander("🔧 Debug - Raw Response", expanded=False):
            st.write("**Response structure:**")
            st.write("Keys:", list(raw_response.keys()) if isinstance(raw_response, dict) else "Not a dict")
            st.json(raw_response)

        # Normalize and extract text
        normalized = normalize_to_claude_like(raw_response)
        extracted_text = extract_text(normalized)

        with st.expander("🔧 Debug - Extracted Text", expanded=False):
            st.write(f"**Extracted text ({len(extracted_text)} chars):**")
            st.code(extracted_text)

            if extracted_text.count('{') != extracted_text.count('}'):
                st.warning("⚠️ JSON có thể bị cắt (unbalanced braces)")
                st.write(f"Open braces: {extracted_text.count('{')}, Close braces: {extracted_text.count('}')}")

        dish: Dish = parse_and_validate(normalized)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.success("✅ JSON hợp lệ theo schema")
            st.json(dish.model_dump(mode="json"))

            with st.expander("🔎 Raw response"):
                st.code(json.dumps(raw_response, ensure_ascii=False, indent=2), language="json")

        with col2:
            _render_metrics(metrics, model_name, extracted_text)

    except Exception as e:
        _render_error(e, raw_response, model_name)


def _render_metrics(metrics: Dict[str, Any], model_name: str, extracted_text: str):
    """Render metrics and performance indicators."""
    st.subheader("📊 Thông số Model")
    st.metric("Model", model_name)
    st.metric("Thời gian xử lý", f"{metrics['latency_s']}s")
    st.metric("Tokens đầu vào", f"{metrics['tokens_in']:,}")
    st.metric("Tokens đầu ra", f"{metrics['tokens_out']:,}")
    st.metric("Chi phí ước tính", f"${metrics['cost_est_usd']:.6f}")

    # Performance indicators
    if metrics['latency_s'] < 2:
        st.success("🚀 Tốc độ: Nhanh")
    elif metrics['latency_s'] < 5:
        st.info("⚡ Tốc độ: Trung bình")
    else:
        st.warning("🐌 Tốc độ: Chậm")


def _render_error(error: Exception, raw_response: Dict[str, Any], model_name: str):
    """Render error information and debugging details."""
    st.error(f"❌ Parse/Validate thất bại: {error}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**🔧 Debug Info:**")
        st.json(raw_response)

    with col2:
        try:
            normalized = normalize_to_claude_like(raw_response)
            extracted = extract_text(normalized)
            st.write(f"**📝 Extracted text ({len(extracted)} chars):**")
            st.code(extracted)

            if extracted.count('{') != extracted.count('}'):
                st.error(f"🔥 JSON bị cắt! Open: {extracted.count('{')}, Close: {extracted.count('}')}")
                st.info("💡 **Giải pháp:** Tăng max_tokens lên 1024-2048")

            try:
                fixed_json = extract_json_from_text(extracted)
                st.success("✅ JSON đã được sửa:")
                st.code(fixed_json)

                parsed = json.loads(fixed_json)
                st.info("🎉 JSON fixed có thể parse được!")

            except Exception as fix_err:
                st.warning(f"❌ Không thể sửa JSON: {fix_err}")

        except Exception as extract_err:
            st.error(f"Cannot extract text: {extract_err}")

