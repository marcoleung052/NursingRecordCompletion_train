# Dockerfile

# 使用一個包含 Python 的基礎映像
FROM python:3.10-slim

# 設置工作目錄
WORKDIR /app

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴，包括 PyTorch 和 Transformers
# 由於 PyTorch 檔案較大，這一步會耗時
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式程式碼
COPY api_server.py .

# 暴露端口 (Hugging Face Spaces 默認使用 7860，但我們使用 $PORT 環境變量)
EXPOSE 7860

# 啟動 Uvicorn 伺服器
# $PORT 是一個由 Hugging Face 自動注入的環境變量，用於指定容器應監聽的端口
# 語法: uvicorn [文件名]:[FastAPI 實例名稱] --host 0.0.0.0 --port $PORT
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"] 
# 雖然實際運行時應使用 $PORT，但在 Dockerfile 中指定一個端口 (如 7860) 是一個常見的做法
# 如果啟動失敗，請嘗試將 "7860" 替換為 "$PORT"
