import os, time, json, io
import streamlit as st
from PIL import Image
from typing import Dict, Any, List

from src.utils import (
    get_logger, MODEL_ID, TEMPERATURE, MAX_TOKENS, MOCK_MODE,
    LIVE_IMAGE_ALWAYS, to_base64
)
# 👇 thêm các builder mới cho Nova & Jamba
from src.prompt_builder import (
    build_prompt, build_prompt_with_image,
    build_prompt_nova, build_prompt_jamba
)
from src.parser import parse_and_validate, extract_text
from src.schema import Dish

logger = get_logger("ui")

st.set_page_config(page_title="Demo LLM → JSON món ăn", page_icon="🍜", layout="wide")
st.title("🍜 Demo: Mô tả món ăn → JSON nguyên liệu")

# Enable auto-rerun for development
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = True

# ====== SIDEBAR ======
st.sidebar.header("⚙️ Cấu hình")
st.sidebar.info("Mock mode hiện đang **BẬT** để phù hợp demo thực tế mà không cần AWS.")
st.sidebar.write(f"MOCK_MODE = {'✅' if MOCK_MODE else '❌'}")

with st.sidebar.expander("🐛 Debug Mode"):
    from src.utils import REGION
    auto_rerun = st.checkbox("Auto-rerun on file change", value=True)
    show_debug_info = st.checkbox("Show debug info", value=False)
    if show_debug_info:
        st.write("**Config:**")
        st.write(f"- MODEL_ID: `{MODEL_ID}`")
        st.write(f"- REGION: `{REGION}`")
        st.write(f"- MOCK_MODE: `{MOCK_MODE}`")
        st.write(f"- LIVE_IMAGE_ALWAYS: `{LIVE_IMAGE_ALWAYS}`")
        st.write("Session state:", st.session_state)

# ========= Client (LIVE) =========
bedrock_client = None
# Khởi tạo client khi cần LIVE (text khi MOCK=false hoặc ảnh khi LIVE_IMAGE_ALWAYS=true)
if (not MOCK_MODE) or LIVE_IMAGE_ALWAYS:
    try:
        from src.bedrock_client import BedrockClient
        bedrock_client = BedrockClient()
    except Exception as e:
        st.sidebar.error(f"Không khởi tạo được Bedrock client (LIVE): {e}")

# ========= Helpers (MOCK) =========
def mock_response_for_text(desc: str) -> Dict[str, Any]:
    # GIỮ NGUYÊN INGREDIENTS BẠN ĐÃ SỬA
    mock_json = {
        "dish_name": "Phở bò" if "phở" in desc.lower() else "Món ăn",
        "cuisine": "Vietnamese",
        "ingredients": [
            {"name": "bánh phở", "quantity": "200 g", "unit": "g"},
            {"name": "thịt bò thăn", "quantity": "250 g", "unit": "g"},
            {"name": "nước dùng bò", "quantity": "500 ml", "unit": "ml"},
            {"name": "hành lá", "quantity": "2 nhánh", "unit": None},
            {"name": "quế", "quantity": "1 thanh", "unit": None},
            {"name": "gừng", "quantity": "1 củ nhỏ", "unit": None},
            {"name": "rau mùi", "quantity": "1 ít", "unit": None}
        ],
        "notes": ["Khẩu phần và định lượng có thể thay đổi theo nhu cầu"]
    }
    return {"content": [{"type": "text", "text": json.dumps(mock_json, ensure_ascii=False)}]}

def mock_response_for_image(desc: str, img: Image.Image) -> Dict[str, Any]:
    # Heuristic nhỏ để demo: nếu ảnh lớn -> thêm 'hành tây'
    w, h = img.size
    extra = {"name": "hành tây", "quantity": "1/2 củ", "unit": None} if max(w, h) > 512 else None
    base = mock_response_for_text(desc)
    data = json.loads(extract_text(base))
    if extra:
        data["ingredients"].append(extra)
    return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}

# ========= Body router & Normalizer =========
def build_body_for_model(model_id: str, desc: str, temperature: float, max_tokens: int) -> dict:
    # Debug logging to see what's being passed
    logger.info(f"Building prompt for model: {model_id}")
    
    mid = (model_id or "").strip().lower()

    # Claude models: check for 'anthropic' or 'claude'
    if 'anthropic' in mid or 'claude' in mid:
        logger.info(f"Using Claude prompt builder for {model_id}")
        return build_prompt(desc, temperature=temperature, max_tokens=max_tokens)
    
    # Nova models: amazon.nova-* 
    if 'nova' in mid:
        logger.info(f"Using Nova prompt builder for {model_id}")
        return build_prompt_nova(desc, temperature=temperature, max_tokens=max_tokens)

    # Jamba 1.5 models: ai21.jamba*
    if 'jamba' in mid or 'ai21' in mid:
        logger.info(f"Using Jamba prompt builder for {model_id}")
        return build_prompt_jamba(desc, temperature=temperature, max_tokens=max_tokens)

    # Fallback to Claude format (not ideal but for safety)
    logger.warning(f"Unknown model type: {model_id}, falling back to Claude format")
    return build_prompt(desc, temperature=temperature, max_tokens=max_tokens)


def normalize_to_claude_like(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bọc response về dạng {"content":[{"type":"text","text":"..."}]} để tái dùng parse_and_validate().
    - Claude: đã đúng -> trả nguyên
    - Nova/Jamba: cố gắng lấy outputText hoặc các field phổ biến -> bọc lại
    """
    if isinstance(raw, dict) and "content" in raw:
        return raw

    txt = None
    if isinstance(raw, dict):
        # Nova/Jamba thường có 'outputText'
        txt = raw.get("outputText")
        if not txt:
            # dò vài key phổ biến khác
            for key in ("result", "output", "completion", "text"):
                val = raw.get(key)
                if isinstance(val, str) and val.strip():
                    txt = val.strip()
                    break
    if not txt:
        # fallback: stringify toàn bộ raw
        txt = json.dumps(raw, ensure_ascii=False)

    return {"content": [{"type": "text", "text": txt}]}

# ========= Invoke =========
def invoke_text(desc: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    if MOCK_MODE:
        return mock_response_for_text(desc)
    if not bedrock_client or not MODEL_ID:
        raise RuntimeError("Bedrock client/MODEL_ID chưa sẵn sàng")
    body = build_body_for_model(MODEL_ID, desc, temperature, max_tokens)
    raw = bedrock_client.invoke(model_id=MODEL_ID, body=body)
    return normalize_to_claude_like(raw)

def invoke_image(desc: str, img: Image.Image, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """
    Ảnh ưu tiên LIVE ngay cả khi text đang MOCK; thiếu credentials -> fallback MOCK để không vỡ demo.
    (Ảnh hiện dùng builder Claude multimodal; nếu sau bạn muốn dùng Nova/Jamba-vision thì đổi router tương ứng)
    """
    if bedrock_client and LIVE_IMAGE_ALWAYS and MODEL_ID:
        try:
            buf = io.BytesIO()
            img_format = "PNG"
            mime = "image/png"
            img.save(buf, img_format)
            b64 = to_base64(buf.getvalue())
            body = build_prompt_with_image(desc, b64, mime, temperature=temperature, max_tokens=max_tokens)
            raw = bedrock_client.invoke(model_id=MODEL_ID, body=body)
            return normalize_to_claude_like(raw)
        except Exception as e:
            st.warning(f"Ảnh LIVE lỗi, fallback MOCK: {e}")
            return mock_response_for_image(desc, img)
    # fallback MOCK
    return mock_response_for_image(desc, img)

# ========= Render =========
def render_result(raw_response: Dict[str, Any], t0: float, mode_label: str):
    try:
        dish: Dish = parse_and_validate(raw_response)
        elapsed = time.time() - t0
        st.success("✅ JSON hợp lệ theo schema")
        st.json(dish.model_dump(mode="json"))
        st.caption(f"⏱️ Thời gian xử lý: {elapsed:.2f}s • Chế độ: {mode_label}")
        with st.expander("🔎 Raw response"):
            st.code(json.dumps(raw_response, ensure_ascii=False, indent=2), language="json")
    except Exception as e:
        st.error(f"❌ Parse/Validate thất bại: {e}")

# ========= Benchmark =========
def benchmark_models(desc: str, models: List[str], temperature: float, max_tokens: int) -> List[Dict[str, Any]]:
    results = []
    for mid in models:
        t0 = time.time()
        try:
            if MOCK_MODE:
                base = mock_response_for_text(desc)
                latency = 0.5 + 0.3 * (hash(mid) % 5) / 5
                time.sleep(min(latency, 1.2))
                text = extract_text(base)
                valid = True
                tokens_in = 120 + (hash(mid) % 50)
                tokens_out = len(text) // 4
                cost_est = round(tokens_in/1e6*0.8 + tokens_out/1e6*1.2, 6)
                results.append({
                    "model": mid,
                    "latency_s": round(time.time()-t0, 2),
                    "valid_rate": 1.0 if valid else 0.0,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_est_usd": cost_est,
                })
            else:
                if not bedrock_client:
                    raise RuntimeError("Bedrock client chưa sẵn sàng")
                # 🔁 chọn builder đúng theo từng model
                body = build_body_for_model(mid, desc, temperature, max_tokens)
                raw = bedrock_client.invoke(model_id=mid, body=body)
                raw_norm = normalize_to_claude_like(raw)
                _ = parse_and_validate(raw_norm)
                results.append({
                    "model": mid,
                    "latency_s": round(time.time()-t0, 2),
                    "valid_rate": 1.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_est_usd": 0.0,
                })
        except Exception as e:
            results.append({
                "model": mid,
                "latency_s": round(time.time()-t0, 2),
                "valid_rate": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_est_usd": 0.0,
                "error": str(e)
            })
    return results

# ========= UI TABS =========
tab_extract, tab_bench = st.tabs(["🍲 Extract JSON", "🏁 Benchmark"])

with tab_extract:
    st.subheader("Chọn chế độ nhập: chỉ **một** trong hai")
    input_mode = st.radio("Input mode", ["Text", "Image"], horizontal=True)

    with st.expander("📋 Hướng dẫn", expanded=True):
        st.markdown(
            "- **Text**: Nhập mô tả món ăn → mô hình trích JSON.\n"
            "- **Image**: Tải ảnh món ăn → mô hình trích JSON **từ ảnh** (không dùng mô tả text).\n"
            "- Chỉ dùng **một** nguồn đầu vào tại một thời điểm."
        )

    # Điều khiển chung
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.number_input("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)
    with c2:
        max_tokens = st.number_input("Max tokens", 64, 4096, MAX_TOKENS, 64)
    with c3:
        run_extract = st.button("Extract Ingredients", type="primary")

    # ===== Mode TEXT =====
    if input_mode == "Text":
        user_desc = st.text_area("Mô tả món ăn", value="Hãy cho tôi nguyên liệu của món phở bò.", height=100)
        img = None

    # ===== Mode IMAGE =====
    else:
        user_desc = ""  # khi ở Image mode, không dùng mô tả
        image_file = st.file_uploader("Tải ảnh món ăn (PNG/JPG/JPEG)", type=["png","jpg","jpeg"])
        img = None
        if image_file is not None:
            img = Image.open(image_file)
            # Hiển thị ảnh thu nhỏ trong expander (mặc định đóng) để UI gọn
            with st.expander("📷 Ảnh đầu vào (thu nhỏ)", expanded=False):
                thumb = img.copy()
                thumb.thumbnail((320, 320))
                st.image(thumb, use_container_width=False)

    if run_extract:
        t0 = time.time()
        try:
            if input_mode == "Text":
                if not user_desc.strip():
                    st.warning("Vui lòng nhập mô tả món ăn.")
                else:
                    raw_response = invoke_text(user_desc, float(temperature), int(max_tokens))
                    mode_label = "MOCK (text)" if MOCK_MODE else "LIVE (text)"
                    render_result(raw_response, t0, mode_label)
            else:
                if img is None:
                    st.warning("Vui lòng tải ảnh món ăn.")
                else:
                    # Ở Image mode: dùng prompt ảnh tăng cường trong build_prompt_with_image
                    raw_response = invoke_image(
                        desc="",  # để build_prompt_with_image tự chèn hướng dẫn mặc định cho ảnh
                        img=img,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                    mode_label = "LIVE (image)" if (bedrock_client and LIVE_IMAGE_ALWAYS and MODEL_ID) else "MOCK (image)"
                    render_result(raw_response, t0, mode_label)
        except Exception as e:
            st.error(f"❌ Lỗi xử lý: {e}")



with tab_bench:
    st.subheader("So sánh nhiều model (giả lập khi MOCK)")
    st.caption("Chọn model → chạy benchmark → xem bảng + biểu đồ. Ở MOCK, số liệu minh hoạ phục vụ báo cáo.")

    available_models = [
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "ai21.jamba-1-5-mini-v1:0",
        "amazon.nova-lite-v1:0",
    ]
    selected_models = st.multiselect("Chọn model", available_models, default=available_models[:2])

    cc1, cc2, _ = st.columns([1,1,2])
    with cc1:
        bench_temp = st.number_input("Temperature (bench)", 0.0, 1.0, float(TEMPERATURE), 0.1)
    with cc2:
        bench_max = st.number_input("Max tokens (bench)", 64, 4096, int(MAX_TOKENS), 64)

    user_desc_bench = st.text_area("Mô tả dùng benchmark", value="Hãy cho tôi nguyên liệu của món phở bò.", height=80)
    run_bench = st.button("Benchmark models")

    if run_bench:
        rows = benchmark_models(user_desc_bench, selected_models, float(bench_temp), int(bench_max))

        tab_overview, tab_charts = st.tabs(["📋 Overview", "📈 Charts"])
        with tab_overview:
            sort_metric = st.selectbox(
                "Sắp xếp theo",
                options=["latency_s", "valid_rate", "cost_est_usd", "tokens_in", "tokens_out", "model"],
                index=0
            )
            ascending = st.checkbox("Sắp xếp tăng dần", value=(sort_metric not in ["valid_rate"]))
            rows_sorted = sorted(rows, key=lambda r: r.get(sort_metric, 0), reverse=not ascending)
            st.dataframe(rows_sorted, use_container_width=True)

        with tab_charts:
            try:
                import matplotlib.pyplot as plt

                labels = [r["model"] for r in rows]
                lat = [r["latency_s"] for r in rows]
                val = [r["valid_rate"] for r in rows]
                cost = [r["cost_est_usd"] for r in rows]

                apply_same_order = st.checkbox("Áp dụng thứ tự sắp xếp giống Overview", value=True)
                if apply_same_order:
                    labels = [r["model"] for r in rows_sorted]
                    lat = [r["latency_s"] for r in rows_sorted]
                    val = [r["valid_rate"] for r in rows_sorted]
                    cost = [r["cost_est_usd"] for r in rows_sorted]

                with st.expander("⏱️ Latency (s)", expanded=True):
                    fig1 = plt.figure()
                    plt.bar(labels, lat)
                    plt.title("Latency (s)")
                    plt.xticks(rotation=15, ha='right')
                    st.pyplot(fig1, use_container_width=True)

                with st.expander("✔️ Valid rate (0-1)", expanded=False):
                    fig2 = plt.figure()
                    plt.bar(labels, val)
                    plt.title("Valid rate (0-1)")
                    plt.xticks(rotation=15, ha='right')
                    st.pyplot(fig2, use_container_width=True)

                with st.expander("💲 Ước tính chi phí (USD)", expanded=False):
                    fig3 = plt.figure()
                    plt.bar(labels, cost)
                    plt.title("Estimated Cost (USD)")
                    plt.xticks(rotation=15, ha='right')
                    st.pyplot(fig3, use_container_width=True)
            except Exception as e:
                st.info(f"Không thể vẽ biểu đồ: {e}")

st.divider()
