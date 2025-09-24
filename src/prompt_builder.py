import json
from .schema import DISH_JSON_SCHEMA


SYSTEM_INSTRUCTIONS = (
    "Bạn là trợ lý ẩm thực chuyên trích xuất nguyên liệu.\n"
    "Yêu cầu: trả về DUY NHẤT một JSON hợp lệ theo schema đã cho.\n"
    "Không trả lời giải thích, không thêm text ngoài JSON.\n"
    "Nếu thông tin không chắc chắn, để trống hoặc bỏ qua trường đó thay vì bịa.\n"
    "Ngôn ngữ ưu tiên: giữ tiếng Việt cho tên nguyên liệu.\n"
)

SYSTEM_INSTRUCTIONS_IMAGE = """
    Bạn là trợ lý ẩm thực chuyên trích xuất nguyên liệu TỪ ẢNH món ăn.
    Yêu cầu: trả về DUY NHẤT một JSON hợp lệ theo schema đã cho.
    Không trả lời giải thích, không thêm text ngoài JSON.
    Nếu thông tin không chắc chắn, để trống hoặc bỏ qua trường đó thay vì bịa.
    Ngôn ngữ ưu tiên: giữ tiếng Việt cho tên nguyên liệu.
    Tránh tên chung chung như "thịt", "rau" - hãy cụ thể như "thịt bò thăn", "rau mùi".
    """


FEW_SHOT_EXAMPLE = {
    "dish_name": "Phở bò",
    "cuisine": "Vietnamese",
    "ingredients": [
        {"name":"bánh phở","quantity":"200","unit":"g"},
        {"name":"thịt bò thăn","quantity":"250","unit":"g"},
        {"name":"hành lá","quantity":"2","unit":"nhánh"},
        {"name":"quế","quantity":"1","unit":"thanh"},
        {"name":"gừng","quantity":"1","unit":"củ"},
        {"name":"rau mùi","quantity":"1","unit":"ít"},
    ],
    "notes":["Định lượng có thể thay đổi theo khẩu vị"]
}


def _build_user_text(user_dish_description: str, include_example: bool = True) -> str:
    """Tạo nội dung text cho user message"""
    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)
    
    user_text = f"""
Nhiệm vụ: Từ mô tả sau, hãy xuất JSON nguyên liệu theo đúng schema.

Mô tả món ăn:
\"\"\"{user_dish_description}\"\"\"

Yêu cầu định dạng:
- Trả về duy nhất một JSON.
- Tuân thủ schema (bên dưới) cả về key và kiểu.
- QUAN TRỌNG: Tách riêng số lượng và đơn vị:
  + "quantity": chỉ chứa SỐ (ví dụ: "200", "1", "2")
  + "unit": chỉ chứa ĐƠN VỊ (ví dụ: "g", "ml", "củ", "nhánh", "quả", "tép")
- Nếu không có đơn vị rõ ràng: unit = null

Schema (JSON Schema):
{schema_str}"""

    if include_example:
        example_str = json.dumps(FEW_SHOT_EXAMPLE, ensure_ascii=False)
        user_text += f"""

Ví dụ (chỉ tham khảo, KHÔNG lẫn vào output):
{example_str}"""

    user_text += """

Lưu ý: Quantity LUÔN là chuỗi số ("1", "200", "0.5"), Unit là chuỗi đơn vị ("g", "ml", "củ") hoặc null.
Trước khi trả: tự kiểm tra JSON hợp lệ theo schema. Nếu chưa hợp lệ, tự sửa rồi mới trả."""

    return user_text.strip()


def build_prompt(user_dish_description: str, temperature: float = 0.2, max_tokens: int = 512):
    """Tạo prompt cho Claude Anthropic"""
    user_text = _build_user_text(user_dish_description)
    
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_INSTRUCTIONS,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def build_prompt_nova(user_dish_description: str, temperature: float = 0.2, max_tokens: int = 512):
    """Tạo prompt cho Amazon Nova"""
    user_text = _build_user_text(user_dish_description)
    
    return {
        "inputText": f"{SYSTEM_INSTRUCTIONS}\n\n{user_text}",
        "inferenceConfig": {
            "temperature": temperature,
            "topP": 0.9,
            "maxTokenCount": max_tokens
        }
    }


def build_prompt_jamba(user_dish_description: str, temperature: float = 0.2, max_tokens: int = 512):
    """Tạo prompt cho AI21 Jamba 1.5"""
    user_text = _build_user_text(user_dish_description)
    
    return {
        "inputText": f"{SYSTEM_INSTRUCTIONS}\n\n{user_text}",
        "inferenceConfig": {
            "temperature": temperature,
            "topP": 0.9,
            "maxTokens": max_tokens  # Some Jamba models use maxTokenCount instead
        }
    }


def build_prompt_with_image(
    user_dish_description: str,
    image_b64: str,
    image_mime: str = "image/png",
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    """Tạo prompt cho Claude Anthropic với ảnh - Two-step approach"""
    schema_str = json.dumps(DISH_JSON_SCHEMA, ensure_ascii=False)
    example_str = json.dumps(FEW_SHOT_EXAMPLE, ensure_ascii=False)

    # Step 1: Identify the dish, Step 2: Generate ingredients as if describing that dish
    user_text = f"""
        Bước 1: Nhìn vào ảnh, xác định tên món ăn cụ thể (ví dụ: "Phở bò", "Bún chả", "Cơm tấm", v.v.)

        Bước 2: Sau khi xác định tên món, hãy tạo JSON nguyên liệu như thể bạn đang trả lời câu hỏi: 
        "Hãy cho tôi nguyên liệu của món [tên món vừa nhận diện]"

        Mô tả bổ sung từ người dùng (nếu có):
        \"\"\"{user_dish_description}\"\"\"

        Yêu cầu định dạng:
        - Trả về duy nhất một JSON.
        - Tuân thủ schema (bên dưới) cả về key và kiểu.
        - QUAN TRỌNG: Tách riêng số lượng và đơn vị:
        + "quantity": chỉ chứa SỐ (ví dụ: "200", "1", "2")
        + "unit": chỉ chứa ĐƠN VỊ (ví dụ: "g", "ml", "củ", "nhánh", "quả", "tép")
        - Nếu không có đơn vị rõ ràng: unit = null

        Schema (JSON Schema):
        {schema_str}

        Ví dụ (chỉ tham khảo, KHÔNG lẫn vào output):
        {example_str}

        Lưu ý: Quantity LUÔN là chuỗi số ("1", "200", "0.5"), Unit là chuỗi đơn vị ("g", "ml", "củ") hoặc null.
        Trước khi trả: tự kiểm tra JSON hợp lệ theo schema. Nếu chưa hợp lệ, tự sửa rồi mới trả.""".strip()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_INSTRUCTIONS_IMAGE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": image_mime, "data": image_b64},
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return body
