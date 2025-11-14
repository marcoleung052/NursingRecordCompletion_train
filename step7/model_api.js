/**
 * model_api.js
 * 呼叫後端代理伺服器，使用 Google Gemini API 進行 Copilot 文本生成。
 */

/*const BACKEND_API_URL = "http://127.0.0.1:5000/predict";*/
const BACKEND_API_URL = "http://120.126.24.45:8001/api/predict";
let apiCallTimeout = null;
let isAPICallInProgress = false;

/**
 * 透過本地後端 API 呼叫 Gemini 進行 Copilot 文本生成。
 * @param {string} inputText - 用戶在輸入框中的全部文字。
 * @returns {Promise<{alternatives: string[]}>} 包含三個紀錄選項的陣列。
 */
async function predictText(inputText) {
    // 限制同時只發起一個 API 請求
    if (isAPICallInProgress) return { alternatives: [] };
    
    const prompt = inputText.trim(); 

    if (prompt.length < 5) { // 至少輸入 5 個字才觸發 AI
        return { alternatives: [] };
    }
    
    const payload = {
        prompt: prompt, 
    };

    try {
        isAPICallInProgress = true;
        
        const response = await fetch(BACKEND_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        isAPICallInProgress = false;

        if (!response.ok) {
            console.error("後端 API 服務錯誤:", response.status);
            return { alternatives: [] };
        }

        // 期望後端返回一個包含三個生成文本的數組：["text1", "text2", "text3"]
        const generatedTexts = responseJson.completions;
        
        if (Array.isArray(generatedTexts) && generatedTexts.length > 0) {
            // 返回時，將第一個結果作為預設建議
            return { 
                alternatives: generatedTexts 
            };
        } else {
            return { alternatives: [] };
        }

    } catch (error) {
        isAPICallInProgress = false;
        console.error("連線錯誤，請確認 Python 後端伺服器 (server.py) 是否運行:", error);
        return { alternatives: [] };
    }
}


/**
 * -----------------------------------------------------------
 * Copilot 模式的觸發器函式 (在主 HTML 腳本中呼叫)
 * -----------------------------------------------------------
 */
function debounceAPICall(inputElement, callback, delay = 10000) {
    if (apiCallTimeout) {
        clearTimeout(apiCallTimeout);
    }
    
    // 清除上一次的預測結果
    callback(null, []); 
    
    if (inputElement.value.trim().length < 5) {
        return;
    }

    // 確保輸入結尾不是空白符號，避免生成無效提示
    const cleanPrompt = inputElement.value.trim();

    apiCallTimeout = setTimeout(async () => {
        try {
            const result = await predictText(cleanPrompt);
            // 成功獲取結果後，調用回調函數
            callback(cleanPrompt, result.alternatives);
        } catch (e) {
            console.error("Debounce API Call Error:", e);
            callback(null, []); // 失敗時清空
        }
    }, delay);
}
