# api_server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# =================================================================
# 1. 應用程式初始化與模型載入
# =================================================================
app = FastAPI(title="GPT-2 Nursing Completion API")

# 設置 CORS：允許前端頁面 (localhost 或您的服務器 IP) 訪問
# ⚠️ 注意：在生產環境中，請將 "http://localhost:5500" 替換為您的前端域名！
origins = [
    "http://localhost:5500",  # 假設您使用 VS Code Live Server 或類似工具
    "http://127.0.0.1:5500",
    "*" # 為了測試方便，暫時允許所有來源
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局變數用於存儲模型和分詞器
tokenizer = None
model = None
MODEL_PATH = "gpt2" # 這裡可以替換為您微調後的模型資料夾路徑

@app.on_event("startup")
async def load_model():
    """在應用啟動時載入 GPT-2 模型"""
    global tokenizer, model
    try:
        # 載入分詞器
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH) 
        
        # 載入預訓練模型或您微調的模型權重
        # 如果您的記憶體允許，可以考慮使用 GPU
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        # model.to(device)
        model.eval() # 設定為評估模式
        
        print(f"✅ GPT-2 模型 {MODEL_PATH} 載入成功！")
    except Exception as e:
        print(f"❌ 模型載入失敗，請檢查 MODEL_PATH 或依賴庫是否安裝：{e}")


# =================================================================
# 2. API 請求與響應格式
# =================================================================
class PredictionRequest(BaseModel):
    """前端發送的請求體格式"""
    prompt: str
    patient_id: str | None = None
    model: str | None = "gpt2-nursing"

class PredictionResponse(BaseModel):
    """後端回傳的響應體格式"""
    completions: list[str]

# =================================================================
# 3. 核心 API 端點 (已修改為生成 3 個序列)
# =================================================================
@app.post("/api/predict", response_model=PredictionResponse)
def predict_completion(request: PredictionRequest):
    """根據輸入提示詞生成 DART 護理紀錄"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="AI 模型服務尚未準備就緒，請檢查後端日誌。")
    
    input_text = request.prompt
    if len(input_text) > 512:
        raise HTTPException(status_code=400, detail="輸入過長，請限制在 512 個字元內。")

    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True)
        
        # 🔥 核心修改：設置 num_return_sequences=3 來生成多個候選結果
        output = model.generate(
            input_ids, 
            max_length=len(input_text) + 150,
            num_return_sequences=3,            # <--- 輸出 3 個不同的補全結果
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id 
        )
        
        all_completions = []
        for sequence in output:
            generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
            
            # 確保內容以用戶的輸入為開頭
            if generated_text.startswith(input_text):
                all_completions.append(generated_text)
            
        # 移除重複的結果並按長度排序
        unique_completions = sorted(list(set(all_completions)), key=len, reverse=True)

        if not unique_completions:
             # 如果模型沒有生成任何有效的補全，則返回用戶輸入本身
             return {"completions": [input_text]}

        # 返回所有唯一的補全結果 (最多 3 個)
        return {"completions": unique_completions}
        
    except Exception as e:
        print(f"推論過程發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"模型推論失敗：{str(e)[:50]}...")

# 運行伺服器
if __name__ == "__main__":
    import uvicorn
    # host 0.0.0.0 允許外部訪問，port 8000 與前端設定一致
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)