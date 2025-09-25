import os, time, json, io
import streamlit as st
from PIL import Image
from typing import Dict, Any, List

from src.utils import (
    get_logger, MODEL_ID, TEMPERATURE, MAX_TOKENS, to_base64
)
from src.prompt_builder import (
    build_prompt, build_prompt_with_image, build_prompt_titan, build_prompt_llama, build_prompt_nova, build_prompt_nova_with_image
)
from src.parser import parse_and_validate, extract_text
from src.schema import Dish

logger = get_logger("ui")

st.set_page_config(page_title="Demo LLM → JSON món ăn", page_icon="🍜", layout="wide")
st.title("🍜 Demo: Mô tả món ăn → JSON nguyên liệu")

# ====== SIDEBAR ======
st.sidebar.header("⚙️ Cấu hình")

with st.sidebar.expander("🐛 Debug Mode"):
    from src.utils import REGION
    show_debug_info = st.checkbox("Show debug info", value=False)
    if show_debug_info:
        st.write("**Config:**")
        st.write(f"- MODEL_ID: `{MODEL_ID}`")
        st.write(f"- REGION: `{REGION}`")
        st.write("Session state:", st.session_state)

# ========= Client (LIVE) =========
bedrock_client = None
try:
    from src.bedrock_client import BedrockClient
    bedrock_client = BedrockClient()
except Exception as e:
    st.sidebar.error(f"Không khởi tạo được Bedrock client: {e}")

# ========= Model definitions =========
IMAGE_MODELS = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
    # Nova Pro không hỗ trợ vision trong Bedrock hiện tại
    # "amazon.nova-pro-v1:0": "Amazon Nova Pro (Vision)",
}

TEXT_MODELS = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
    "amazon.titan-text-premier-v1:0": "Titan Text Premier", 
    "meta.llama3-8b-instruct-v1:0": "Llama 3 8B",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": "Claude 3.5 Haiku",
}


# ========= Body router & Normalizer =========
def build_body_for_model(model_id: str, desc: str, temperature: float, max_tokens: int) -> dict:
    """Tạo prompt body cho model cụ thể"""
    logger.info(f"Building prompt for model: {model_id}")
    
    mid = (model_id or "").strip().lower()

    # Nova models
    if 'nova' in mid:
        logger.info(f"Using Nova prompt builder for {model_id}")
        return build_prompt_nova(desc, temperature=temperature, max_tokens=max_tokens)
    
    # Titan models
    if 'titan' in mid:
        logger.info(f"Using Titan prompt builder for {model_id}")
        return build_prompt_titan(desc, temperature=temperature, max_tokens=max_tokens)
    
    # Llama models
    if 'llama' in mid:
        logger.info(f"Using Llama prompt builder for {model_id}")
        return build_prompt_llama(desc, temperature=temperature, max_tokens=max_tokens)
    
    # Claude models (default)
    logger.info(f"Using Claude prompt builder for {model_id}")
    return build_prompt(desc, temperature=temperature, max_tokens=max_tokens)


def extract_json_from_text(text: str) -> str:
    """
    Trích xuất JSON từ text, handle trường hợp bị cắt ở giữa
    """
    text = text.strip()
    
    # Tìm JSON block đầu tiên
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("Không tìm thấy JSON trong response")
    
    # Tìm end brace matching
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
        # JSON bị cắt - thử fix
        logger.warning("JSON appears to be truncated, attempting to fix...")
        json_text = text[start_idx:]
        
        # Thêm closing braces cho các arrays/objects bị thiếu
        open_braces = json_text.count('{') - json_text.count('}')
        open_brackets = json_text.count('[') - json_text.count(']')
        
        # Đóng arrays trước
        if open_brackets > 0:
            json_text += ']' * open_brackets
            
        # Đóng objects
        if open_braces > 0:
            json_text += '}' * open_braces
            
        return json_text
    
    return text[start_idx:end_idx]


def normalize_to_claude_like(raw: Dict[str, Any]) -> Dict[str, Any]:

    logger.info(f"normalize_to_claude_like input: {type(raw)} - {raw}")

    if isinstance(raw, dict) and "content" in raw:
        return raw

    txt = None
    if isinstance(raw, dict):
        logger.info(f"Raw is dict with keys: {list(raw.keys())}")

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

        # Jamba/Nova text-only: outputText ở root
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

        # Các key phổ biến khác
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



# ========= Invoke =========
def invoke_model(desc: str, model_id: str, temperature: float, max_tokens: int, img: Image.Image = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Invoke model và trả về (response, metrics)
    metrics chứa: latency_s, tokens_in, tokens_out, cost_est_usd
    """
    if not bedrock_client or not model_id:
        raise RuntimeError("Bedrock client/model_id chưa sẵn sàng")
    
    t0 = time.time()
    
    if img is not None:
        # Image processing - hỗ trợ Claude và Nova
        buf = io.BytesIO()
        img_format = "PNG"
        mime = "image/png"
        img.save(buf, img_format)
        b64 = to_base64(buf.getvalue())
        
        # Choose appropriate image builder based on model
        if 'nova' in model_id.lower():
            body = build_prompt_nova_with_image(desc, b64, mime, temperature=temperature, max_tokens=max_tokens)
        else:
            # Default to Claude format for image processing
            body = build_prompt_with_image(desc, b64, mime, temperature=temperature, max_tokens=max_tokens)
    else:
        # Text processing
        body = build_body_for_model(model_id, desc, temperature, max_tokens)
    
    raw = bedrock_client.invoke(model_id=model_id, body=body)
    latency = time.time() - t0
    
    # Log raw response để debug
    logger.info(f"Raw response type: {type(raw)}")
    logger.info(f"Raw response: {raw}")
    
    # Kiểm tra nếu response rỗng hoặc None
    if not raw:
        logger.error("Response từ Bedrock là rỗng!")
        raise RuntimeError("AWS Bedrock trả về response rỗng")
    
    # Estimate metrics (simplified)
    tokens_in = len(desc.split()) * 1.3  # rough estimate
    response_text = extract_text(normalize_to_claude_like(raw))
    tokens_out = len(response_text.split()) * 1.3
    
    # Cost estimation (rough, model-dependent)
    cost_per_1k_in = 0.003 if 'claude' in model_id.lower() else 0.001
    cost_per_1k_out = 0.015 if 'claude' in model_id.lower() else 0.002
    cost_est = (tokens_in/1000 * cost_per_1k_in) + (tokens_out/1000 * cost_per_1k_out)
    
    metrics = {
        "latency_s": round(latency, 2),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out), 
        "cost_est_usd": round(cost_est, 6)
    }
    
    return normalize_to_claude_like(raw), metrics

# ========= Render =========
def render_result(raw_response: Dict[str, Any], metrics: Dict[str, Any], model_name: str):
    """Render kết quả với 2 columns layout"""
    try:
        # Debug info trước khi parse
        with st.expander("🔧 Debug - Raw Response", expanded=False):
            st.write("**Response structure:**")
            st.write("Keys:", list(raw_response.keys()) if isinstance(raw_response, dict) else "Not a dict")
            st.json(raw_response)
            
        # Normalize và extract text
        normalized = normalize_to_claude_like(raw_response)
        extracted_text = extract_text(normalized)
        
        with st.expander("🔧 Debug - Extracted Text", expanded=False):
            st.write(f"**Extracted text ({len(extracted_text)} chars):**")
            st.code(extracted_text)
            
            # Check if JSON appears truncated
            if extracted_text.count('{') != extracted_text.count('}'):
                st.warning("⚠️ JSON có thể bị cắt (unbalanced braces)")
                st.write(f"Open braces: {extracted_text.count('{')}, Close braces: {extracted_text.count('}')}")
        
        dish: Dish = parse_and_validate(normalized)
        
        # Two column layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success("✅ JSON hợp lệ theo schema")
            st.json(dish.model_dump(mode="json"))
            
            with st.expander("🔎 Raw response"):
                st.code(json.dumps(raw_response, ensure_ascii=False, indent=2), language="json")
        
        with col2:
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
            
            # Response quality indicators
            response_length = len(extracted_text)
            if response_length > 900:
                st.info("📄 Response dài - chất lượng cao")
            elif response_length < 300:
                st.warning("📝 Response ngắn - có thể thiếu thông tin")
                
            # Model-specific tips
            if 'titan-premier' in model_name.lower():
                st.info("💡 Titan Premier: Tốt cho response phức tạp")
            elif 'claude' in model_name.lower():
                st.info("🎯 Claude: Chính xác cao, hiểu context tốt")
            elif 'llama' in model_name.lower():
                st.info("🦙 Llama: Mã nguồn mở, cost-effective")
                
    except Exception as e:
        st.error(f"❌ Parse/Validate thất bại: {e}")
        
        # Enhanced error info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔧 Debug Info:**")
            st.json(raw_response)
        
        with col2:
            # Try to show extracted text and diagnose issue
            try:
                normalized = normalize_to_claude_like(raw_response)
                extracted = extract_text(normalized)
                st.write(f"**📝 Extracted text ({len(extracted)} chars):**")
                st.code(extracted)
                
                # JSON validation check
                if extracted.count('{') != extracted.count('}'):
                    st.error(f"🔥 JSON bị cắt! Open: {extracted.count('{')}, Close: {extracted.count('}')}")
                    st.info("💡 **Giải pháp:** Tăng max_tokens lên 1024-2048")
                
                # Try to extract and show partial JSON
                try:
                    fixed_json = extract_json_from_text(extracted)
                    st.success("✅ JSON đã được sửa:")
                    st.code(fixed_json)
                    
                    # Try to parse the fixed JSON
                    parsed = json.loads(fixed_json)
                    st.info("🎉 JSON fixed có thể parse được!")
                    
                except Exception as fix_err:
                    st.warning(f"❌ Không thể sửa JSON: {fix_err}")
                    
            except Exception as extract_err:
                st.error(f"Cannot extract text: {extract_err}")
        
        # Response length analysis
        try:
            response_length = len(extract_text(normalize_to_claude_like(raw_response)))
            if response_length > 800:
                st.info("ℹ️ **Response dài** - rất có thể bị cắt do max_tokens quá thấp")
            if 'titan-text-premier' in model_name.lower():
                st.info("💡 **Titan Premier** thường cần max_tokens >= 1024")
        except:
            pass



# ========= MAIN UI =========
st.subheader("🍲 Trích xuất nguyên liệu từ mô tả hoặc hình ảnh")

# Input mode selection
input_mode = st.radio("Chọn chế độ nhập:", ["Text", "Image"], horizontal=True)

# Model selection based on input mode
if input_mode == "Text":
    selected_model_id = st.selectbox(
        "Chọn model cho văn bản:",
        options=list(TEXT_MODELS.keys()),
        format_func=lambda x: TEXT_MODELS[x],
        index=0
    )
    model_name = TEXT_MODELS[selected_model_id]
else:
    selected_model_id = st.selectbox(
        "Chọn model cho hình ảnh:",
        options=list(IMAGE_MODELS.keys()),
        format_func=lambda x: IMAGE_MODELS[x],
        index=0
    )
    model_name = IMAGE_MODELS[selected_model_id]

# Common controls
col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.number_input("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)
with col2:
    # Tăng default max_tokens cho Titan Premier và Claude Sonnet
    default_max_tokens = MAX_TOKENS
    if 'titan-text-premier' in selected_model_id:
        default_max_tokens = 1024
    elif 'claude-3-5-sonnet' in selected_model_id:
        default_max_tokens = 1024
    max_tokens = st.number_input("Max tokens", 64, 4096, default_max_tokens, 64)
with col3:
    run_extract = st.button("Extract Ingredients", type="primary")

# Input based on mode
if input_mode == "Text":
    user_desc = st.text_area("Mô tả món ăn", value="Hãy cho tôi nguyên liệu của món phở bò.", height=100)
    img = None
else:
    user_desc = ""
    image_file = st.file_uploader("Tải ảnh món ăn (PNG/JPG/JPEG)", type=["png","jpg","jpeg"])
    img = None
    if image_file is not None:
        img = Image.open(image_file)
        with st.expander("📷 Ảnh đầu vào", expanded=False):
            thumb = img.copy()
            thumb.thumbnail((320, 320))
            st.image(thumb, use_container_width=False)

# Process extraction
if run_extract:
    try:
        if input_mode == "Text" and not user_desc.strip():
            st.warning("Vui lòng nhập mô tả món ăn.")
        elif input_mode == "Image" and img is None:
            st.warning("Vui lòng tải ảnh món ăn.")
        else:
            with st.spinner(f"Đang xử lý với {model_name}..."):
                if input_mode == "Text":
                    raw_response, metrics = invoke_model(
                        user_desc, selected_model_id, float(temperature), int(max_tokens)
                    )
                else:
                    raw_response, metrics = invoke_model(
                        "", selected_model_id, float(temperature), int(max_tokens), img
                    )
                
                render_result(raw_response, metrics, model_name)
                
    except Exception as e:
        st.error(f"❌ Lỗi xử lý: {e}")

st.divider()
st.caption("� Powered by AWS Bedrock • Hỗ trợ đa model AI")