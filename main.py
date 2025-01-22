from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

app = FastAPI()

# Конфигурация YandexGPT
FOLDER_ID = os.getenv("FOLDER_ID", "b1g2715hdfor5i697ugv")  # Ваш folder_id по умолчанию
API_KEY = os.getenv("API_KEY")  # Будет загружен из переменных окружения
API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"status": "API is running"}

@app.post("/check_text")
async def check_text(request: TextRequest):
    try:
        headers = {
            "Authorization": f"Api-Key {API_KEY}",
            "x-folder-id": FOLDER_ID
        }

        system_prompt = """Ты корректор русского языка. ВАЖНО:
1. ТОЛЬКО исправляй орфографические и грамматические ошибки
2. НИКОГДА не меняй содержание текста
3. НИКОГДА не генерируй новый текст
4. НИКОГДА не добавляй новые слова или фразы
5. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО менять знаки препинания:
   - Сохранять все точки с запятой (;)
   - Сохранять все двоеточия (:)
   - Сохранять все маркеры списков
6. Возвращай ТОЧНО тот же текст с теми же знаками препинания
7. ЗАПРЕЩЕНО менять структуру текста и списков
8. Сохранять все переносы строк и отступы
9. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО заменять букву 'е' на 'ё'
10. НИКОГДА не менять регистр в следующих случаях:
    - Слово 'Общество' в контексте организации (ООО, АО)
    - Названия компаний и организаций
    - Любые слова в кавычках, являющиеся названиями
11. СТРОГО сохранять регистр в следующих названиях:
    - 'Что делать Аудит'
    - 'Что делать Консалт'
12. Сохранять заглавные буквы в названиях организаций"""

        payload = {
            "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "temperature": 0.1,
                "maxTokens": 15000,
                "stream": False
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": request.text}
            ]
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return {"result": result["result"]["alternatives"][0]["message"]["text"]}
        else:
            raise HTTPException(status_code=500, detail="Error from YandexGPT API")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
