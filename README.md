# 🍜 Food Ingredient Extractor

Ứng dụng trích xuất nguyên liệu từ mô tả món ăn hoặc hình ảnh, trả về JSON có cấu trúc sử dụng AWS Bedrock AI.

## 🤖 Supported Models

### Text Models
- **Claude 3.5 Sonnet** - Anthropic's flagship model
- **Titan Text G1 - Lite** - Amazon's lightweight model (4K tokens, cost-effective)
- **Titan Text G1 - Express** - Amazon's balanced model (8K tokens, good performance)
- **Llama 3 8B** - Meta's open-source model
- **Claude 3.5 Haiku** - Anthropic's fast model

### Image Models  
- **Claude 3.5 Sonnet** - Text + Image understanding
- **Amazon Nova Pro** - Multimodal model
- **Amazon Nova Lite** - Lightweight multimodal model

## 🆕 Model Updates

**Latest Change:** Replaced Titan Text Premier with two new models:
- **Titan Text G1 - Lite** (`amazon.titan-text-lite-v1`) - Optimized for cost and fine-tuning
- **Titan Text G1 - Express** (`amazon.titan-text-express-v1`) - Balanced performance and RAG support

## 🚀 Chạy nhanh

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

## 📋 Cách sử dụng

### Trích xuất từ văn bản
1. Chọn tab "🍲 Extract JSON"
2. Chọn "Text" 
3. Nhập mô tả món ăn (VD: "Nguyên liệu làm phở bò")
4. Click "Extract Ingredients"

### Trích xuất từ hình ảnh
1. Chọn tab "🍲 Extract JSON"  
2. Chọn "Image"
3. Tải lên ảnh món ăn
4. Click "Extract Ingredients"

## ⚙️ Cấu hình

App mặc định chạy ở **MOCK MODE** (không cần AWS). 

Để dùng AWS Bedrock thật:
1. Tạo file `.env`:
```
MOCK_MODE=false
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

## 📄 Kết quả

JSON output:
```json
{
  "dish_name": "Phở bò",
  "cuisine": "Vietnamese", 
  "ingredients": [
    {"name": "bánh phở", "quantity": "200", "unit": "g"},
    {"name": "thịt bò", "quantity": "250", "unit": "g"}
  ]
}
```
